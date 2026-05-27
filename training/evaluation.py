"""
Robust evaluation suite for benchmarking the Relational Transformer-PPO Agent 
against fixed heuristic baselines in multi-player configurations.
Synchronized for Multi-Dispatch (N-vector) relational reasoning.
"""
import argparse
import os
import sys
import math
import numpy as np
import torch

# Enforce clean path insertions for local package lookups
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kaggle_environments import make
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline

class EvaluationTournament:
    """ Manages structured tournament evaluation for a trained policy checkpoint. """
    def __init__(self, agent_path: str, max_entities: int = 200, device: str = 'cpu'):
        self.device = device
        self.max_entities = max_entities
        
        # 1. Initialize environment processing layers matching training standards
        self.wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
        self.wrapper = OrbitWarsWrapper(self.wrapper_config)
        self.obs_proc = ObservationProcessor(max_entities=max_entities, board_size=self.wrapper_config["boardSize"], max_speed=self.wrapper_config["shipSpeed"])
        self.act_proc = ActionProcessor(self.wrapper)
        
        # 2. Instantiate and load relational model architecture
        self.model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=max_entities)
        if os.path.exists(agent_path):
            self.model.load_state_dict(torch.load(agent_path, map_location=device))
            print(f"Loaded tournament candidate weights from: {agent_path}")
        else:
            print(f"⚠️ Checkpoint not found at {agent_path}. Evaluating uninitialized network.")
            
        self.model.to(device)
        self.model.eval()

    def get_agent_actions(self, p_obs: dict, raw_obs: dict, player_id: int) -> list:
        """ Runs a deterministic argmax inference pass over valid relational multi-dispatch pairs. """
        # Construct turn-by-turn action mask grid matrix [N, N]
        mask_vector = self.wrapper.get_action_mask(raw_obs, player_id=player_id, allocation_percentage=1.0)
        action_masks_grid = np.zeros((self.max_entities, self.max_entities), dtype=bool)
        
        planets_raw = raw_obs.get("planets", [])
        planets_count = len(planets_raw)
        for s_idx in range(planets_count):
            if planets_raw[s_idx][1] == player_id:
                action_masks_grid[s_idx, :planets_count] = mask_vector

        # Forward pass under deterministic inference constraints
        with torch.no_grad():
            entities_t = torch.tensor(p_obs['entities'], dtype=torch.float32).unsqueeze(0).to(self.device)
            entity_ids_t = torch.tensor(p_obs['entity_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            mask_t = torch.tensor(p_obs['mask'], dtype=torch.float32).unsqueeze(0).to(self.device)
            act_masks_t = torch.tensor(action_masks_grid, dtype=torch.bool).unsqueeze(0).to(self.device)

            # target_logits shape: [1, N, N], alloc_logits shape: [1, N, N, 6]
            target_logits, alloc_logits, _ = self.model(entities_t, entity_ids_t, mask_t, act_masks_t)
            
            # Deterministic argmax across the target dimension for every source node [N]
            sampled_targets = target_logits.squeeze(0).argmax(dim=-1)
            
            # Deterministic argmax for allocations along the chosen target paths
            batch_idx = torch.arange(self.max_entities, device=self.device)
            selected_alloc_logits = alloc_logits.squeeze(0)[batch_idx, sampled_targets, :]
            sampled_allocs = selected_alloc_logits.argmax(dim=-1)

        # Build owned_indices mask to filter learner moves
        is_source_owned = (p_obs['entities'][:, 2] == 1.0)
        valid_source_mask = is_source_owned & (p_obs['mask'] == 1.0)
        owned_indices = np.where(valid_source_mask)[0]
        
        learner_target_indices = sampled_targets.cpu().numpy()[owned_indices].tolist()
        learner_alloc_indices = sampled_allocs.cpu().numpy()[owned_indices].tolist()

        # Translate grid matrices back into continuous Kaggle-compatible vectors
        return self.act_proc.process_actions(
            raw_obs, player_id=player_id,
            target_indices=learner_target_indices,
            allocation_indices=learner_alloc_indices
        )

    def run_tournament(self, num_games: int = 20) -> dict:
        """ Runs evaluation matchups while cycling player slot indices to enforce spatial invariance. """
        win_count = 0
        placement_history = []
        
        print(f"Starting evaluation tournament over {num_games} rounds...")

        for game_idx in range(num_games):
            # Rotate target agent seat position across 0, 1, 2, 3 to ensure slot robust generalization
            my_slot = game_idx % 4
            
            env = make("orbit_wars", debug=False)
            obs_list = env.reset()
            
            # Instantiate baseline models for all slots
            bots = [HeuristicBaseline(pid) for pid in range(4)]
            
            done = False
            step = 0
            
            while not done and step < 500:
                actions = [[] for _ in range(4)]
                
                # Step each player slot appropriately based on active rotation state
                for pid in range(4):
                    raw_obs = obs_list[pid]['observation']
                    if pid == my_slot:
                        # Process candidate model action multi-dispatch grid
                        p_obs = self.obs_proc.process(raw_obs, player_id=my_slot)
                        actions[my_slot] = self.get_agent_actions(p_obs, raw_obs, player_id=my_slot)
                    else:
                        # Process static heuristic choice
                        actions[pid] = bots[pid].act(raw_obs)
                        
                obs_list = env.step(actions)
                done = any(state.get('status') != 'ACTIVE' for state in obs_list)
                step += 1

            # Compute terminal ranking profiles from game outcomes
            final_scores = [state.get('reward', 0) if state.get('reward') is not None else 0 for state in obs_list]
            candidate_score = final_scores[my_slot]
            
            # Sort scores in descending order to isolate placement rank
            sorted_scores = sorted(final_scores, reverse=True)
            placement = sorted_scores.index(candidate_score) + 1
            placement_history.append(placement)
            
            if placement == 1:
                win_count += 1
                
            print(f"Round {game_idx+1:02d}/{num_games:02d} -> Slot: {my_slot} | Final Scores: {final_scores} | Placement: #{placement}")

        win_rate = (win_count / num_games) * 100.0
        avg_placement = np.mean(placement_history)
        
        results = {"win_rate": win_rate, "avg_placement": avg_placement}
        print(f"\n🏆 Tournament Benchmark Summary:\nCandidate Win Rate: {win_rate:.2f}%\nAverage Match Placement: #{avg_placement:.2f}")
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, default='checkpoints/bc_pretrained.pt')
    parser.add_argument('--num_games', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tournament = EvaluationTournament(agent_path=args.agent_path, device=args.device)
    tournament.run_tournament(num_games=args.num_games)
