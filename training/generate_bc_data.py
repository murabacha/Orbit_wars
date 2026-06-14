"""
Final Audited BC Data Collection (Dual-Head Ready).
Buffers episodes to calculate final Win/Loss returns for Critic Pre-training.
"""
import argparse
import math
import os
import numpy as np
from kaggle_environments import make

from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline
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
        returns=np.array(returns, dtype=np.float32) # NEW: Critic Targets!
    )

def collect_rollouts(num_transitions: int, save_path: str, max_entities: int = 200, checkpoint_interval: int = 500):
    wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    entities_list, entity_ids_list, mask_list, target_list, alloc_list, returns_list = [], [], [], [], [], []
    collected = 0
    
    if os.path.exists(save_path):
        try:
            with np.load(save_path) as data:
                entities_list, entity_ids_list, mask_list = list(data['entities']), list(data['entity_ids']), list(data['mask'])
                target_list, alloc_list = list(data['target']), list(data['alloc'])
                # Handle backwards compatibility if returns aren't in old dataset
                returns_list = list(data['returns']) if 'returns' in data else [0.0] * len(target_list)
            collected = len(entities_list)
            print(f"✅ Successfully resumed from {collected}/{num_transitions}")
        except: print("⚠️ Starting fresh.")

    env = make('orbit_wars', debug=False)
    episode = 0
    
    while collected < num_transitions:
        episode += 1
        obs_list = env.reset()
        num_players = len(obs_list)
        baselines = [HeuristicBaseline(pid) for pid in range(num_players)]
        obs_procs = [ObservationProcessor(max_entities=max_entities, board_size=100.0, max_speed=6.0) for _ in range(num_players)]
        done, steps = False, 0
        
        # EPISODE BUFFER: Store data temporarily until game ends
        ep_data = {pid: {'entities': [], 'ids': [], 'masks': [], 'targets': [], 'allocs': []} for pid in range(num_players)}

        while not done and steps < 500:
            actions = [baselines[pid].act(obs_list[pid]['observation']) if obs_list[pid]['status'] == 'ACTIVE' else [] for pid in range(num_players)]
            for pid in range(num_players):
                if obs_list[pid]['status'] != 'ACTIVE': continue
                player_obs, player_moves = obs_list[pid]['observation'], actions[pid]
                
                if player_moves:
                    processed = obs_procs[pid].process(player_obs, player_id=pid)
                    planets = player_obs.get('planets', [])
                    raw_entity_ids = [p[0] for p in planets] + [f[0] for f in player_obs.get('fleets', [])]
                    raw_entity_ids = raw_entity_ids[:processed['entity_ids'].shape[0]]
                    step_targets, step_allocs, has_valid_move = np.zeros(max_entities, dtype=np.int64), np.zeros(max_entities, dtype=np.int64), False
                    
                    for m in player_moves:
                        source_id, heuristic_angle, ships_to_send = m
                        source_planet = next((p for p in planets if p[0] == source_id), None)
                        if source_planet is None or source_id not in raw_entity_ids: continue
                        s_index = raw_entity_ids.index(source_id)
                        src_x, src_y, src_rad, src_ships = source_planet[2], source_planet[3], source_planet[4], source_planet[5]
                        target_id, best_diff = None, float('inf')
                        
                        for p in planets:
                            if p[0] == source_id: continue
                            p_id, p_owner, px, py, p_rad, p_ships, p_prod = p[:7]
                            tgt_model = {'x': px, 'y': py, 'radius': p_rad, 'id': p_id, 'owner': p_owner, 'production': p_prod, 'ships': p_ships, 'source_ships': src_ships}
                            solver_angle, _, _, _ = wrapper.get_intercept_params((src_x, src_y), src_rad, tgt_model, ships_to_send/src_ships if src_ships > 0 else 0, player_obs)
                            diff = abs(((solver_angle - heuristic_angle + math.pi) % (2 * math.pi)) - math.pi)
                            if diff < best_diff: best_diff, target_id = diff, p_id
                            
                        if target_id is not None and target_id in raw_entity_ids and best_diff <= 0.25:
                            step_targets[s_index], step_allocs[s_index], has_valid_move = raw_entity_ids.index(target_id), alloc_to_index(ships_to_send, src_ships), True
                    
                    if has_valid_move:
                        # Append to Episode Buffer, not global list
                        ep_data[pid]['entities'].append(processed['entities'])
                        ep_data[pid]['ids'].append(processed['entity_ids'])
                        ep_data[pid]['masks'].append(processed['mask'])
                        ep_data[pid]['targets'].append(step_targets)
                        ep_data[pid]['allocs'].append(step_allocs)

            obs_list = env.step(actions)
            done = all(s.get('status') != 'ACTIVE' for s in obs_list)
            steps += 1
            
        # --- GAME OVER: Assign Terminal Rewards ---
        ep_matched = 0
        for pid in range(num_players):
            if len(ep_data[pid]['entities']) == 0: continue
            
            # Calculate Win/Loss using identical logic to rewards.py
            final_obs = obs_list[pid]['observation']
            planets = final_obs.get("planets", [])
            fleets = final_obs.get("fleets", [])
            
            my_ships = sum(p[5] for p in planets if p[1] == pid) + sum(f[4] for f in fleets if f[1] == pid)
            enemy_ships = sum(p[5] for p in planets if p[1] not in [pid, -1]) + sum(f[4] for f in fleets if f[1] not in [pid, -1])
            
            # PPO Terminal Target Alignment (+5.0 Win, -5.0 Loss)
            ep_return = 5.0 if my_ships > enemy_ships else -5.0
            
            # Push buffered data to global lists
            L = len(ep_data[pid]['entities'])
            entities_list.extend(ep_data[pid]['entities'])
            entity_ids_list.extend(ep_data[pid]['ids'])
            mask_list.extend(ep_data[pid]['masks'])
            target_list.extend(ep_data[pid]['targets'])
            alloc_list.extend(ep_data[pid]['allocs'])
            returns_list.extend([ep_return] * L) # Tag every frame with the outcome!
            
            collected += L
            ep_matched += L
            
        print(f"Episode {episode} finished. Extracted {ep_matched} frames. Total: {collected}/{num_transitions}")
        if collected % checkpoint_interval < 500: # Approximate saving
            save_dataset(save_path, entities_list, entity_ids_list, mask_list, target_list, alloc_list, returns_list)

    save_dataset(save_path, entities_list, entity_ids_list, mask_list, target_list, alloc_list, returns_list)
    print(f"Finalized dataset: {collected} rows.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_transitions', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--max_entities', type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    args = parser.parse_args()
    collect_rollouts(args.num_transitions, args.save_path, max_entities=args.max_entities, checkpoint_interval=args.checkpoint_interval)
