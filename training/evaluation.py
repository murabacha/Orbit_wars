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
from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel, load_checkpoint_with_surgery
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline

class EvaluationTournament:
    """ Manages structured tournament evaluation for a trained policy checkpoint. """
    def __init__(self, agent_path: str, max_entities: int = 200, device: str = 'cpu'):
        self.device = device
        self.max_entities = max_entities
        
        self.wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
        self.wrapper = OrbitWarsWrapper(self.wrapper_config)
        self.obs_proc = ObservationProcessor(max_entities=max_entities, board_size=self.wrapper_config["boardSize"], max_speed=self.wrapper_config["shipSpeed"])
        self.act_proc = ActionProcessor(self.wrapper)
        
        self.model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=max_entities)
        if os.path.exists(agent_path):
            load_checkpoint_with_surgery(self.model, agent_path, device)
            print(f"Loaded tournament candidate weights from: {agent_path}")
        else:
            print(f"⚠️ Checkpoint not found at {agent_path}. Evaluating uninitialized network.")
            
        self.model.to(device)
        self.model.eval()

    def get_agent_actions(self, p_obs: dict, raw_obs: dict, player_id: int, min_ships: float = 1.0) -> list:
        # FIX: Use 2D pairwise mask directly from wrapper
        full_mask = self.wrapper.get_action_mask(raw_obs, player_id=player_id)
        action_masks_grid = np.zeros((self.max_entities, self.max_entities), dtype=bool)
        action_masks_grid[:full_mask.shape[0], :full_mask.shape[1]] = full_mask

        with torch.no_grad():
            ent_t = torch.tensor(p_obs['entities'], dtype=torch.float32).unsqueeze(0).to(self.device)
            ids_t = torch.tensor(p_obs['entity_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            msk_t = torch.tensor(p_obs['mask'], dtype=torch.float32).unsqueeze(0).to(self.device)
            amsk_t = torch.tensor(action_masks_grid, dtype=torch.bool).unsqueeze(0).to(self.device)

            target_logits, alloc_logits, _ = self.model(ent_t, ids_t, msk_t, amsk_t)
            sampled_targets = target_logits.squeeze(0).argmax(dim=-1)
            
            batch_idx = torch.arange(self.max_entities, device=self.device)
            selected_alloc_logits = alloc_logits.squeeze(0)[batch_idx, sampled_targets, :]

            # --- THE SMART VOLUME MASK (101 Vectorized Bins) ---
            source_ships = (ent_t.squeeze(0)[:, 9] * 1000.0)
            target_is_owned = (ent_t.squeeze(0)[sampled_targets, 2] == 1.0)
            target_ships = (ent_t.squeeze(0)[sampled_targets, 9] * 1000.0)
            is_attack = ~target_is_owned
            
            # For bins 0-75 (absolute counts):
            sends_abs = torch.arange(76, device=self.device).unsqueeze(0).expand(source_ships.shape[0], -1).float()
            sends_abs = torch.min(sends_abs, source_ships.unsqueeze(1))
            # For bins 76-100 (percentages):
            pcts = torch.linspace(0.0, 1.0, 25, device=self.device)
            sends_pct = (source_ships.unsqueeze(1) * pcts).int().float()
            sends = torch.cat([sends_abs, sends_pct], dim=1) # Shape: (N, 101)
            
            is_attack_val = is_attack.unsqueeze(1)
            target_ships_val = target_ships.unsqueeze(1)
            
            trickle_mask = (sends < min_ships) & (~is_attack_val | (sends <= target_ships_val))
            trickle_mask[:, 0] = False
            
            # THE SHUFFLE MASK: Block useless intra-empire shuffling for 1% to 99%
            trickle_mask[target_is_owned, 1:100] = True
            
            selected_alloc_logits[trickle_mask] = -1e9
            # ----------------------------------------------------
            
            sampled_allocs = selected_alloc_logits.argmax(dim=-1)

        # PASS FULL LISTS: ActionProcessor now uses the planet's actual index (i) 
        # to look up target/alloc, so we MUST provide the full mapping.
        return self.act_proc.process_actions(
            raw_obs, player_id=player_id,
            target_indices=sampled_targets.cpu().numpy().tolist(),
            allocation_indices=sampled_allocs.cpu().numpy().tolist()
        )

    def run_tournament(self, num_games: int = 20) -> dict:
        win_count = 0
        placement_history = []
        
        print(f"Starting evaluation tournament over {num_games} rounds...")

        for game_idx in range(num_games):
            env = make("orbit_wars", debug=False)
            obs_list = env.reset()
            num_players = len(obs_list)
            
            my_slot = game_idx % num_players
            bots = [HeuristicBaseline(pid) for pid in range(num_players)]
            
            done = False
            step = 0
            while not done and step < 500:
                actions = [[] for _ in range(num_players)]
                for pid in range(num_players):
                    if obs_list[pid]['status'] != 'ACTIVE':
                        continue
                        
                    raw_obs = obs_list[pid]['observation']
                    if pid == my_slot:
                        p_obs = self.obs_proc.process(raw_obs, player_id=my_slot)
                        actions[my_slot] = self.get_agent_actions(p_obs, raw_obs, player_id=my_slot)
                    else:
                        actions[pid] = bots[pid].act(raw_obs)
                        
                obs_list = env.step(actions)
                done = (obs_list[my_slot].get('status') != 'ACTIVE') or all(s['status'] != 'ACTIVE' for i, s in enumerate(obs_list) if i != my_slot)
                step += 1

            final_scores = [state.get('reward', 0) if state.get('reward') is not None else 0 for state in obs_list]
            candidate_score = final_scores[my_slot]
            sorted_scores = sorted(final_scores, reverse=True)
            placement = sorted_scores.index(candidate_score) + 1
            placement_history.append(placement)
            
            if placement == 1:
                win_count += 1
            print(f"Round {game_idx+1:02d}/{num_games:02d} -> Slot: {my_slot} (Players: {num_players}) | Final Scores: {final_scores} | Placement: #{placement}")

        win_rate = (win_count / num_games) * 100.0
        avg_placement = np.mean(placement_history)
        print(f"\n🏆 Tournament Benchmark Summary:\nCandidate Win Rate: {win_rate:.2f}%\nAverage Match Placement: #{avg_placement:.2f}")
        return {"win_rate": win_rate, "avg_placement": avg_placement}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, default='checkpoints/bc_pretrained.pt')
    parser.add_argument('--num_games', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tournament = EvaluationTournament(agent_path=args.agent_path, device=args.device)
    tournament.run_tournament(num_games=args.num_games)
