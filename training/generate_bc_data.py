"""
Generate Behavioral Cloning dataset by rolling out the HeuristicBaseline.
Each sample contains the processed observation (entities, entity_ids, mask)
and the discrete actions inferred from the heuristic (target_index, allocation_index).

Saves data as a compressed numpy .npz file with arrays:
 - entities: (N, max_entities, feature_dim)
 - entity_ids: (N, max_entities)
 - mask: (N, max_entities)
 - target: (N,)
 - alloc: (N,)

Notes:
 - Mapping heuristic moves (source_id, angle, ships) to discrete target index is done
   by finding the planet whose bearing from the source best matches the heuristic angle.
 - Allocation index maps ship fraction to bins: [0, 0.25, 0.5, 0.75, 1.0, exact_needed]

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


def nearest_entity_by_angle(source, planets, angle):
    best_idx = None
    best_diff = float('inf')
    sx, sy = source.x, source.y
    for p in planets:
        if p.id == source.id:
            continue
        a = angle_between(sx, sy, p.x, p.y)
        diff = abs(((a - angle + math.pi) % (2 * math.pi)) - math.pi)
        if diff < best_diff:
            best_diff = diff
            best_idx = p
    return best_idx


def alloc_to_index(ships_to_send, source_ships):
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
    # If it doesn't match common bins closely, set to "exact_needed" (index 5)
    if best_diff > 0.15:
        return 5
    return idx


def collect_rollouts(num_transitions, save_path, max_entities=200):
    obs_proc = ObservationProcessor(max_entities=max_entities)
    wrapper = OrbitWarsWrapper({})
    env = make('orbit_wars', debug=False)

    entities_list = []
    entity_ids_list = []
    mask_list = []
    target_list = []
    alloc_list = []

    collected = 0
    episode = 0
    print(f"Starting BC data collection target={num_transitions}")

    while collected < num_transitions:
        episode += 1

        # Start a new environment for this episode
        env = make('orbit_wars', debug=False)
        obs_list = env.reset()
        num_players = len(obs_list)
        baselines = [HeuristicBaseline(pid) for pid in range(num_players)]
        done = False
        steps = 0

        while not done and collected < num_transitions and steps < 1000:
            # Create actions for every player
            actions = []
            for pid in range(num_players):
                current_obs = obs_list[pid]['observation']
                action = baselines[pid].act(current_obs)
                actions.append(action)

            # Capture player 0 observation before stepping
            current_obs = obs_list[0]['observation']
            move = baselines[0].act(current_obs)

            if move:
                processed = obs_proc.process(current_obs, 0)
                planets = [p for p in current_obs.get('planets', [])]
                raw_entity_ids = [p[0] for p in current_obs.get('planets', [])] + [f[0] for f in current_obs.get('fleets', [])]
                raw_entity_ids = raw_entity_ids[:processed['entity_ids'].shape[0]]

                for m in move:
                    try:
                        source_id, angle, ships_to_send = m
                    except Exception:
                        continue
                    source_planet = next((p for p in planets if p[0] == source_id), None)
                    if source_planet is None:
                        continue
                    sx, sy = source_planet[2], source_planet[3]

                    target_id = None
                    best_diff = float('inf')
                    for p in planets:
                        if p[0] == source_id:
                            continue
                        tx, ty = p[2], p[3]
                        a = angle_between(sx, sy, tx, ty)
                        diff = abs(((a - angle + math.pi) % (2 * math.pi)) - math.pi)
                        if diff < best_diff:
                            best_diff = diff
                            target_id = p[0]

                    if target_id is None:
                        continue
                    if target_id not in raw_entity_ids:
                        continue
                    t_index = raw_entity_ids.index(target_id)
                    alloc_idx = alloc_to_index(ships_to_send, source_planet[5])
                    entities_list.append(processed['entities'])
                    entity_ids_list.append(processed['entity_ids'])
                    mask_list.append(processed['mask'])
                    target_list.append(t_index)
                    alloc_list.append(alloc_idx)

                    collected += 1
                    if collected % 1000 == 0:
                        print(f"Collected {collected} samples")
                    if collected >= num_transitions:
                        break

            obs_list = env.step(actions)
            done = any(state.get('status') != 'ACTIVE' for state in obs_list)
            steps += 1

        print(f"Episode {episode} finished, collected so far: {collected}")

    # Convert lists to arrays
    entities_arr = np.array(entities_list, dtype=np.float32)
    entity_ids_arr = np.array(entity_ids_list, dtype=np.int64)
    mask_arr = np.array(mask_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.int64)
    alloc_arr = np.array(alloc_list, dtype=np.int64)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(save_path, entities=entities_arr, entity_ids=entity_ids_arr, mask=mask_arr, target=target_arr, alloc=alloc_arr)
    print(f"Saved BC dataset to {save_path} with {collected} samples")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_transitions', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--max_entities', type=int, default=200)
    args = parser.parse_args()
    collect_rollouts(args.num_transitions, args.save_path, max_entities=args.max_entities)
