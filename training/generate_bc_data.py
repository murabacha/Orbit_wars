"""
Diverse Multi-Agent BC Data Collection (Dual-Head Ready).
Pools multiple Kaggle agents with different strategies to collect rich tactical data.
Buffers episodes to calculate final Win/Loss returns for Critic Pre-training.
"""
import argparse
import math
import os
import random
import numpy as np
from kaggle_environments import make

from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper

def alloc_to_index(ships_to_send: int, source_ships: int) -> int:
    if source_ships <= 0: return 0
    frac = ships_to_send / float(source_ships)
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    idx, best_diff = 0, float('inf')
    for i, b in enumerate(bins):
        d = abs(frac - b)
        if d < best_diff:
            best_diff, idx = d, i
    return 5 if best_diff > 0.12 else idx

def save_dataset(save_path, entities, ids, masks, targets, allocs, returns):
    if not entities: return
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(
        save_path, 
        entities=np.array(entities, dtype=np.float32), 
        entity_ids=np.array(ids, dtype=np.int64), 
        mask=np.array(masks, dtype=np.float32), 
        target=np.array(targets, dtype=np.int64), 
        alloc=np.array(allocs, dtype=np.int64),
        returns=np.array(returns, dtype=np.float32)
    )

def collect_rollouts(num_transitions: int, save_path: str, max_entities: int = 200, checkpoint_interval: int = 2000):
    wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    entities_list, entity_ids_list, mask_list, target_list, alloc_list, returns_list = [], [], [], [], [], []
    collected = 0
    
    # AGENT POOL
    # Use full absolute paths to ensure kaggle_environments can load them
    base_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
    AGENT_POOL = [
        os.path.join(base_dir, "agents/baseline/elite_heuristic.py"),
        os.path.join(base_dir, "online_agents/agent_1/main.py"),
        os.path.join(base_dir, "online_agents/agent_2/main.py"),
        os.path.join(base_dir, "online_agents/agent_3/main.py"),
        os.path.join(base_dir, "online_agents/an_agent.py")
    ]
    
    # Filter out missing agents
    AGENT_POOL = [a for a in AGENT_POOL if os.path.exists(a)]
    print(f"🤖 Agent Pool initialized with {len(AGENT_POOL)} agents.")

    if os.path.exists(save_path):
        try:
            with np.load(save_path) as data:
                entities_list = list(data['entities'])
                entity_ids_list = list(data['entity_ids'])
                mask_list = list(data['mask'])
                target_list = list(data['target'])
                alloc_list = list(data['alloc'])
                returns_list = list(data['returns'])
            collected = len(entities_list)
            print(f"✅ Successfully resumed from {collected}/{num_transitions}")
        except: print("⚠️ Starting fresh.")

    env = make('orbit_wars', debug=False)
    episode = 0
    
    while collected < num_transitions:
        episode += 1
        num_players = random.choice([2, 4])
        selected_agents = [random.choice(AGENT_POOL) for _ in range(num_players)]
        
        print(f"🎬 Episode {episode}: {num_players} players. Agents: {[os.path.basename(a) for a in selected_agents]}")
        
        # Run the match
        try:
            env.run(selected_agents)
        except Exception as e:
            print(f"❌ Match failed: {e}. Skipping episode.")
            continue
        
        # EPISODE BUFFER: Store data temporarily until we know the outcome
        ep_data = {pid: {'entities': [], 'ids': [], 'masks': [], 'targets': [], 'allocs': []} for pid in range(num_players)}
        obs_procs = [ObservationProcessor(max_entities=max_entities, board_size=100.0, max_speed=6.0) for _ in range(num_players)]
        
        # Process steps
        for i in range(len(env.steps) - 1):
            for pid in range(num_players):
                state_i = env.steps[i][pid]
                action_i = env.steps[i+1][pid].get('action')
                
                if state_i['status'] != 'ACTIVE': continue
                
                player_obs = state_i['observation']
                if action_i and len(action_i) > 0:
                    processed = obs_procs[pid].process(player_obs, player_id=pid)
                    planets = player_obs.get('planets', [])
                    raw_entity_ids = [p[0] for p in planets] + [f[0] for f in player_obs.get('fleets', [])]
                    raw_entity_ids = raw_entity_ids[:processed['entity_ids'].shape[0]]
                    
                    step_targets = np.zeros(max_entities, dtype=np.int64)
                    step_allocs = np.zeros(max_entities, dtype=np.int64)
                    has_valid_move = False
                    
                    for m in action_i:
                        if not (isinstance(m, list) and len(m) == 3): continue
                        source_id, heuristic_angle, ships_to_send = m
                        
                        source_planet = next((p for p in planets if p[0] == source_id), None)
                        if source_planet is None or source_id not in raw_entity_ids: continue
                        
                        s_index = raw_entity_ids.index(source_id)
                        src_x, src_y, src_rad, src_ships = source_planet[2], source_planet[3], source_planet[4], source_planet[5]
                        
                        target_id, best_diff = None, float('inf')
                        for p in planets:
                            if p[0] == source_id: continue
                            p_id, p_owner, px, py, p_rad, p_ships, p_prod = p[:7]
                            frac = ships_to_send / src_ships if src_ships > 0 else 0
                            tgt_model = {'x': px, 'y': py, 'radius': p_rad, 'id': p_id, 'owner': p_owner, 'production': p_prod, 'ships': p_ships, 'source_ships': src_ships}
                            
                            solver_angle, _, _, _ = wrapper.get_intercept_params((src_x, src_y), src_rad, tgt_model, frac, player_obs)
                            diff = abs(((solver_angle - heuristic_angle + math.pi) % (2 * math.pi)) - math.pi)
                            if diff < best_diff:
                                best_diff = diff
                                target_id = p_id
                        
                        if target_id is not None and target_id in raw_entity_ids and best_diff <= 0.25:
                            t_index = raw_entity_ids.index(target_id)
                            step_targets[s_index] = t_index
                            step_allocs[s_index] = alloc_to_index(ships_to_send, src_ships)
                            has_valid_move = True
                            
                    if has_valid_move:
                        ep_data[pid]['entities'].append(processed['entities'])
                        ep_data[pid]['ids'].append(processed['entity_ids'])
                        ep_data[pid]['masks'].append(processed['mask'])
                        ep_data[pid]['targets'].append(step_targets)
                        ep_data[pid]['allocs'].append(step_allocs)

        # Assign Terminal Returns
        ep_extracted = 0
        final_state = env.steps[-1]
        all_rewards = [s.get('reward', 0) for s in final_state]
        max_reward = max(all_rewards)
        
        for pid in range(num_players):
            if len(ep_data[pid]['entities']) == 0: continue
            
            player_reward = final_state[pid].get('reward', 0)
            is_winner = player_reward > 0 and player_reward == max_reward
            ep_return = 5.0 if is_winner else -5.0
            
            L = len(ep_data[pid]['entities'])
            entities_list.extend(ep_data[pid]['entities'])
            entity_ids_list.extend(ep_data[pid]['ids'])
            mask_list.extend(ep_data[pid]['masks'])
            target_list.extend(ep_data[pid]['targets'])
            alloc_list.extend(ep_data[pid]['allocs'])
            returns_list.extend([ep_return] * L)
            
            collected += L
            ep_extracted += L
            
        print(f"✅ Extracted {ep_extracted} frames. Total: {collected}/{num_transitions}")
        
        if episode % 10 == 0:
            save_dataset(save_path, entities_list, entity_ids_list, mask_list, target_list, alloc_list, returns_list)
            print(f"💾 Checkpoint saved: {collected} rows.")

    save_dataset(save_path, entities_list, entity_ids_list, mask_list, target_list, alloc_list, returns_list)
    print(f"🎉 Finalized dataset: {collected} rows.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_transitions', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--max_entities', type=int, default=200)
    args = parser.parse_args()
    collect_rollouts(args.num_transitions, args.save_path, max_entities=args.max_entities)
