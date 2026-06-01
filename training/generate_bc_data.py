"""
Generate Behavioral Cloning dataset by rolling out the HeuristicBaseline.
Includes Resumability and Periodic Checkpointing to protect against Colab disconnects.
"""
import argparse
import math
import os
import random
import numpy as np
from kaggle_environments import make

from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper

def angle_between(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def alloc_to_index(ships_to_send: int, source_ships: int) -> int:
    """Maps continuous ship counts to discrete structural policy target indices."""
    if source_ships <= 0:
        return 0
    frac = ships_to_send / float(source_ships)
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    idx = 0
    best_diff = float('inf')
    for i, b in enumerate(bins):
        d = abs(frac - b)
        if d < best_diff:
            best_diff = d
            idx = i
    if best_diff > 0.12:
        return 5
    return idx

def save_dataset(save_path, entities, ids, masks, targets, allocs):
    """Helper to save the dataset to disk."""
    if not entities: return
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(
        save_path, 
        entities=np.array(entities, dtype=np.float32), 
        entity_ids=np.array(ids, dtype=np.int64), 
        mask=np.array(masks, dtype=np.float32), 
        target=np.array(targets, dtype=np.int64), 
        alloc=np.array(allocs, dtype=np.int64)
    )

def collect_rollouts(num_transitions: int, save_path: str, max_entities: int = 200, checkpoint_interval: int = 500):
    wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    
    entities_list, entity_ids_list, mask_list, target_list, alloc_list = [], [], [], [], []
    collected = 0
    
    if os.path.exists(save_path):
        print(f"📦 Found existing dataset at {save_path}. Loading to resume...")
        try:
            with np.load(save_path) as data:
                entities_list = list(data['entities'])
                entity_ids_list = list(data['entity_ids'])
                mask_list = list(data['mask'])
                target_list = list(data['target'])
                alloc_list = list(data['alloc'])
            collected = len(entities_list)
            print(f"✅ Successfully resumed! Starting from {collected}/{num_transitions} transitions.")
        except Exception as e:
            print(f"⚠️ Could not load existing dataset ({e}). Starting fresh.")

    episode = 0
    print(f"Starting synchronized BC data collection pipeline. Target: {num_transitions} transitions.")

    while collected < num_transitions:
        episode += 1
        ep_seed = random.randint(0, 1000000)
        env = make('orbit_wars', configuration={'seed': ep_seed}, debug=False)
        wrapper = OrbitWarsWrapper(wrapper_config)
        
        obs_list = env.reset()
        num_players = len(obs_list)
        baselines = [HeuristicBaseline(pid) for pid in range(num_players)]
        obs_procs = [ObservationProcessor(max_entities=max_entities, board_size=100.0, max_speed=6.0) for _ in range(num_players)]
        
        done = False
        steps = 0
        ep_collected = 0

        while not done and collected < num_transitions and steps < 500:
            actions = []
            for pid in range(num_players):
                if obs_list[pid]['status'] != 'ACTIVE':
                    actions.append([])
                    continue
                actions.append(baselines[pid].act(obs_list[pid]['observation']))

            for pid in range(num_players):
                if obs_list[pid]['status'] != 'ACTIVE': continue
                player_obs = obs_list[pid]['observation']
                player_moves = actions[pid]

                if player_moves:
                    processed = obs_procs[pid].process(player_obs, player_id=pid)
                    planets = player_obs.get('planets', [])
                    raw_entity_ids = [p[0] for p in planets] + [f[0] for f in player_obs.get('fleets', [])]
                    raw_entity_ids = raw_entity_ids[:processed['entity_ids'].shape[0]]

                    step_targets = np.zeros(max_entities, dtype=np.int64)
                    step_allocs = np.zeros(max_entities, dtype=np.int64)
                    has_valid_move = False

                    for m in player_moves:
                        try:
                            source_id, heuristic_angle, ships_to_send = m
                        except: continue
                            
                        source_planet = next((p for p in planets if p[0] == source_id), None)
                        if source_planet is None or source_id not in raw_entity_ids: continue
                        
                        s_index = raw_entity_ids.index(source_id)
                        src_x, src_y, src_ships = source_planet[2], source_planet[3], source_planet[4]
                        target_id, best_diff = None, float('inf')
                        
                        for p in planets:
                            if p[0] == source_id: continue
                            p_id, _, px, py, p_ships, p_radius, p_prod, *extra = p
                            tgt_model = {'x': px, 'y': py, 'id': p_id, 'owner': p[1], 'production': p_prod, 'ships': p_ships, 'source_ships': src_ships, 'angular_velocity': extra[0] if extra else 0.0}
                            solver_angle, _, _, _ = wrapper.get_intercept_params((src_x, src_y), tgt_model, ships_to_send/src_ships if src_ships > 0 else 0, player_obs)
                            diff = abs(((solver_angle - heuristic_angle + math.pi) % (2 * math.pi)) - math.pi)
                            if diff < best_diff:
                                best_diff, target_id = diff, p_id

                        if target_id is not None and target_id in raw_entity_ids and best_diff <= 0.25:
                            step_targets[s_index] = raw_entity_ids.index(target_id)
                            step_allocs[s_index] = alloc_to_index(ships_to_send, src_ships)
                            has_valid_move = True

                    if has_valid_move:
                        entities_list.append(processed['entities'])
                        entity_ids_list.append(processed['entity_ids'])
                        mask_list.append(processed['mask'])
                        target_list.append(step_targets)
                        alloc_list.append(step_allocs)
                        collected += 1
                        ep_collected += 1
                        
                        # PRECISE CHECKPOINT: Save every N transitions
                        if collected % checkpoint_interval == 0:
                            print(f"💾 Saving periodic checkpoint ({collected}/{num_transitions})...")
                            save_dataset(save_path, entities_list, entity_ids_list, mask_list, target_list, alloc_list)
                            
                        if collected >= num_transitions: break

            obs_list = env.step(actions)
            done = all(state.get('status') != 'ACTIVE' for state in obs_list)
            steps += 1

        print(f"Episode {episode} (Seed {ep_seed}) terminated after {steps} steps. Added {ep_collected} transitions. Total: {collected}")

    save_dataset(save_path, entities_list, entity_ids_list, mask_list, target_list, alloc_list)
    print(f"Successfully finalized BC dataset export to {save_path} with {collected} rows.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_transitions', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--max_entities', type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='Save every N transitions')
    args = parser.parse_args()
    collect_rollouts(args.num_transitions, args.save_path, max_entities=args.max_entities, checkpoint_interval=args.checkpoint_interval)
