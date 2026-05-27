"""
Generate Behavioral Cloning dataset by rolling out the HeuristicBaseline.
Fully synchronized with the predictive tokenized ObservationProcessor and OrbitWarsWrapper contracts.
"""
import argparse
import math
import os
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
    # Allocation bins: [0%, 25%, 50%, 75%, 100%, exact_needed]
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    idx = 0
    best_diff = float('inf')
    for i, b in enumerate(bins):
        d = abs(frac - b)
        if d < best_diff:
            best_diff = d
            idx = i
            
    # If the commitment ratio drifts past common buckets, assign to head 5 (exact_needed)
    if best_diff > 0.12:
        return 5
    return idx

def collect_rollouts(num_transitions: int, save_path: str, max_entities: int = 200):
    # Synchronize and initialize environment physics interfaces
    wrapper_config = {
        "shipSpeed": 6.0,
        "sunRadius": 10.0,
        "boardSize": 100.0,
        "episodeSteps": 500
    }
    wrapper = OrbitWarsWrapper(wrapper_config)
    obs_proc = ObservationProcessor(
        max_entities=max_entities, 
        board_size=wrapper_config["boardSize"], 
        max_speed=wrapper_config["shipSpeed"]
    )

    entities_list = []
    entity_ids_list = []
    mask_list = []
    target_list = []
    alloc_list = []

    collected = 0
    episode = 0
    print(f"Starting synchronized BC data collection pipeline. Target: {num_transitions} transitions.")

    while collected < num_transitions:
        episode += 1
        env = make('orbit_wars', debug=False)
        obs_list = env.reset()
        num_players = len(obs_list)
        baselines = [HeuristicBaseline(pid) for pid in range(num_players)]
        
        done = False
        steps = 0

        while not done and collected < num_transitions and steps < 500:
            # 1. Gather all actions simultaneously to ensure single iteration validation
            actions = []
            for pid in range(num_players):
                current_obs = obs_list[pid]['observation']
                action = baselines[pid].act(current_obs)
                actions.append(action)

            # Isolate player 0 metrics for target behavior compilation
            p0_obs = obs_list[0]['observation']
            p0_moves = actions[0]  # Grab the identical move array executed in env

            if p0_moves:
                # Compile predictive feature observations
                processed = obs_proc.process(p0_obs, player_id=0)
                
                # Dynamic mapping references from updated state structures
                planets = p0_obs.get('planets', [])
                
                # Build target map following observation tokens sorting sequence
                # We assume ObservationProcessor follows this order
                raw_entity_ids = [p[0] for p in planets] + [f[0] for f in p0_obs.get('fleets', [])]
                raw_entity_ids = raw_entity_ids[:processed['entity_ids'].shape[0]]

                for m in p0_moves:
                    try:
                        source_id, heuristic_angle, ships_to_send = m
                    except (ValueError, TypeError):
                        continue
                        
                    # Find source planet using index 4 for current ship totals (as per prompt)
                    source_planet = next((p for p in planets if p[0] == source_id), None)
                    if source_planet is None:
                        continue
                    
                    src_x, src_y = source_planet[2], source_planet[3]
                    src_ships = source_planet[4]  # Verified index position 4 for ships count

                    # Match heuristic firing angle with wrapper intercept math to identify intended token ID
                    target_id = None
                    best_diff = float('inf')
                    
                    for p in planets:
                        if p[0] == source_id:
                            continue
                        
                        # Updated Planet structure: [id, owner, x, y, ships, radius, production, ...]
                        p_id, _, px, py, p_ships, p_radius, p_prod, *extra = p
                        p_w = extra[0] if extra else 0.0
                        
                        tgt_model = {
                            'x': px, 'y': py, 'id': p_id, 'owner': p[1], 
                            'production': p_prod, 'ships': p_ships, 
                            'source_ships': src_ships, 'angular_velocity': p_w
                        }

                        # Compute what angle the wrapper expects to reach this target
                        # Pass ships_to_send / src_ships as allocation_percentage
                        alloc_pct = ships_to_send / src_ships if src_ships > 0 else 0
                        solver_angle, _, _, _ = wrapper.get_intercept_params((src_x, src_y), tgt_model, alloc_pct, p0_obs)
                        
                        # Evaluate angle difference
                        diff = abs(((solver_angle - heuristic_angle + math.pi) % (2 * math.pi)) - math.pi)
                        if diff < best_diff:
                            best_diff = diff
                            target_id = p_id

                    # Only append trajectories where angle match falls within precision tolerances
                    if target_id is None or target_id not in raw_entity_ids:
                        continue
                    if best_diff > 0.25:  # Tolerance ceiling to catch radical physics exceptions
                        continue

                    t_index = raw_entity_ids.index(target_id)
                    alloc_idx = alloc_to_index(ships_to_send, src_ships)

                    # Append metrics to data arrays
                    entities_list.append(processed['entities'])
                    entity_ids_list.append(processed['entity_ids'])
                    mask_list.append(processed['mask'])
                    target_list.append(t_index)
                    alloc_list.append(alloc_idx)

                    collected += 1
                    if collected % 5000 == 0:
                        print(f"Collected {collected}/{num_transitions} samples.")
                    if collected >= num_transitions:
                        break

            # Advance the simulation step cleanly
            obs_list = env.step(actions)
            done = any(state.get('status') != 'ACTIVE' for state in obs_list)
            steps += 1

        print(f"Episode {episode} terminated. Total data collected: {collected}")

    # Compress compiled frames to storage arrays
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(
        save_path, 
        entities=np.array(entities_list, dtype=np.float32), 
        entity_ids=np.array(entity_ids_list, dtype=np.int64), 
        mask=np.array(mask_list, dtype=np.float32), 
        target=np.array(target_list, dtype=np.int64), 
        alloc=np.array(alloc_list, dtype=np.int64)
    )
    print(f"Successfully finalized BC dataset export to {save_path} with {collected} rows.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_transitions', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--max_entities', type=int, default=200)
    args = parser.parse_args()
    collect_rollouts(args.num_transitions, args.save_path, max_entities=args.max_entities)
