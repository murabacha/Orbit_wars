"""
Production-Grade Proximal Policy Optimization (PPO) Training Loop for Orbit Wars AI.
Refactored for Scalar N*N Relational Action Space and Predictive PBRS Rewards.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kaggle_environments import make

# Enforce clean path insertions for local package lookups
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.rewards import RewardShaper
from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, 
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """ Computes Generalized Advantage Estimations (GAE) with terminal bootstrap handling. """
    advantages = np.zeros_like(rewards)
    lastgaelam = 0.0
    
    for t in reversed(range(len(rewards))):
        # nonterminal is 0.0 on the final step of an episode
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam
        
    returns = advantages + values[:-1]
    return advantages, returns


def evaluate_policy_distribution(model: nn.Module, entities: torch.Tensor, entity_ids: torch.Tensor, 
                                 mask: torch.Tensor, action_masks: torch.Tensor, 
                                 target_actions: torch.Tensor, alloc_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes joint log-probabilities and entropy for flattened scalar actions [B].
    Ensures action masking is applied during the PPO update loop.
    """
    # Forward pass returns target_logits [B, N*N] and alloc_logits [B, N*N, 6]
    target_logits, alloc_logits, values = model(entities, entity_ids, mask, action_masks)
    
    # Target distribution over flattened N*N space
    target_dist = torch.distributions.Categorical(logits=target_logits)
    
    # Extract allocation logits matching the chosen target trajectory
    batch_idx = torch.arange(target_actions.shape[0], device=entities.device)
    selected_alloc_logits = alloc_logits[batch_idx, target_actions, :]
    alloc_dist = torch.distributions.Categorical(logits=selected_alloc_logits)
    
    # Compute log probabilities for the taken actions
    log_p_targets = target_dist.log_prob(target_actions)
    log_p_allocs = alloc_dist.log_prob(alloc_actions)
    joint_log_prob = log_p_targets + log_p_allocs
    
    # Entropy calculation for exploration regularization
    entropy = (target_dist.entropy() + alloc_dist.entropy()).mean()
    
    return joint_log_prob, entropy, values.squeeze(-1)


def ppo_update(model: nn.Module, optimizer: optim.Optimizer, rollout_data: dict, 
               config: dict, epochs: int = 4, batch_size: int = 64):
    """ Executes Trust-Region Clipped Surrogate gradient optimization cycles. """
    obs_batch = rollout_data['obs']
    action_targets = np.array(rollout_data['targets']) # Shape: [Total_Steps]
    action_allocs = np.array(rollout_data['allocs'])   # Shape: [Total_Steps]
    old_log_probs = np.array(rollout_data['log_probs']) # Shape: [Total_Steps]
    returns = np.array(rollout_data['returns'])         # Shape: [Total_Steps]
    advantages = np.array(rollout_data['advantages'])   # Shape: [Total_Steps]
    
    dataset_size = len(obs_batch)
    inds = np.arange(dataset_size)
    
    # Standardize advantages to reduce variance
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    device = next(model.parameters()).device

    for _ in range(epochs):
        np.random.shuffle(inds)
        for start in range(0, dataset_size, batch_size):
            mb_inds = inds[start:start + batch_size]
            if len(mb_inds) < batch_size: continue

            # Construct dynamic parallel batch tensors
            entities = torch.tensor(np.stack([o['entities'] for o in obs_batch[mb_inds]]), dtype=torch.float32).to(device)
            entity_ids = torch.tensor(np.stack([o['entity_ids'] for o in obs_batch[mb_inds]]), dtype=torch.long).to(device)
            mask = torch.tensor(np.stack([o['mask'] for o in obs_batch[mb_inds]]), dtype=torch.float32).to(device)
            act_masks = torch.tensor(np.stack([o['action_masks'] for o in obs_batch[mb_inds]]), dtype=torch.bool).to(device)

            targets_t = torch.tensor(action_targets[mb_inds], dtype=torch.long).to(device)
            allocs_t = torch.tensor(action_allocs[mb_inds], dtype=torch.long).to(device)
            old_logp_t = torch.tensor(old_log_probs[mb_inds], dtype=torch.float32).to(device)
            returns_t = torch.tensor(returns[mb_inds], dtype=torch.float32).to(device)
            advantages_t = torch.tensor(advantages[mb_inds], dtype=torch.float32).to(device)

            # Re-evaluate distribution with masks applied to logits
            logp, entropy, values = evaluate_policy_distribution(
                model, entities, entity_ids, mask, act_masks, targets_t, allocs_t
            )

            # PPO Clipped Objective
            ratio = torch.exp(logp - old_logp_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - config['clip_range'], 1.0 + config['clip_range']) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # MSE Value Loss
            value_loss = F.mse_loss(values, returns_t)

            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def train(args):
    config = {
        "player_id": 0,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 2.5e-4,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_entities": 200,
        "n_epochs": 4
    }
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=config["max_entities"])
    if args.bc_checkpoint and os.path.exists(args.bc_checkpoint):
        model.load_state_dict(torch.load(args.bc_checkpoint, map_location=device))
        print(f"Loaded BC checkpoint from: {args.bc_checkpoint}")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    obs_proc = ObservationProcessor(max_entities=config["max_entities"], board_size=wrapper_config["boardSize"], max_speed=wrapper_config["shipSpeed"])
    act_proc = ActionProcessor(wrapper)
    reward_shaper = RewardShaper(player_id=config["player_id"], gamma=config["gamma"], total_training_steps=args.total_timesteps)

    print(f"Relational PPO Fine-Tuning initiated on: {device}")
    total_steps = 0
    episode = 0

    obs_buffer, targets_buffer, allocs_buffer = [], [], []
    log_probs_buffer, values_buffer, rewards_buffer, dones_buffer = [], [], [], []

    while total_steps < args.total_timesteps:
        episode += 1
        env = make("orbit_wars", debug=False)
        obs_list = env.reset()
        
        ep_obs, ep_targets, ep_allocs = [], [], []
        ep_logp, ep_values, ep_rewards, ep_dones = [], [], [], []

        done = False
        steps_counter = 0
        baselines = [HeuristicBaseline(pid) for pid in range(4)]

        while not done and steps_counter < 500:
            actions = [baselines[pid].act(obs_list[pid]['observation']) for pid in range(1, 4)]

            p0_obs = obs_list[0]['observation']
            processed = obs_proc.process(p0_obs, player_id=config["player_id"])
            
            # Action Masking Grid Construction [N, N]
            mask_vector = wrapper.get_action_mask(p0_obs, player_id=config["player_id"], allocation_percentage=1.0)
            action_masks_grid = np.zeros((config["max_entities"], config["max_entities"]), dtype=bool)
            planets_count = len(p0_obs.get("planets", []))
            for s_idx in range(planets_count):
                if p0_obs["planets"][s_idx][1] == config["player_id"]:
                    action_masks_grid[s_idx, :planets_count] = mask_vector

            model.eval()
            with torch.no_grad():
                ent_t = torch.tensor(processed['entities'], dtype=torch.float32).unsqueeze(0).to(device)
                ids_t = torch.tensor(processed['entity_ids'], dtype=torch.long).unsqueeze(0).to(device)
                msk_t = torch.tensor(processed['mask'], dtype=torch.float32).unsqueeze(0).to(device)
                amsk_t = torch.tensor(action_masks_grid, dtype=torch.bool).unsqueeze(0).to(device)

                # target_logits is [1, N*N]
                target_logits, alloc_logits, value_t = model(ent_t, ids_t, msk_t, amsk_t)
                
                target_dist = torch.distributions.Categorical(logits=target_logits.squeeze(0))
                sampled_target_flat = target_dist.sample() # Scalar index in [0, N*N-1]
                
                # Allocation slice strictly for chosen flattened target
                target_idx_val = sampled_target_flat.item()
                selected_alloc_logits = alloc_logits.squeeze(0)[target_idx_val, :]
                alloc_dist = torch.distributions.Categorical(logits=selected_alloc_logits)
                sampled_alloc = alloc_dist.sample()
                
                log_prob = (target_dist.log_prob(sampled_target_flat) + alloc_dist.log_prob(sampled_alloc)).item()

            # Decode flattened index for ActionProcessor
            N = config["max_entities"]
            source_idx = target_idx_val // N
            target_idx = target_idx_val % N

            learner_moves = act_proc.process_actions(
                p0_obs, player_id=config["player_id"], 
                target_indices=[target_idx], 
                allocation_indices=[sampled_alloc.item()]
            )
            # Override act_proc source logic since we already chose the source
            # But wait, act_proc iterates through ALL owned planets.
            # If we sample only one scalar, we should only send from ONE planet.
            # We need to pass the specific source planet ID.
            # Refactoring act_proc slightly for scalar dispatch:
            planets = p0_obs.get("planets", [])
            src_p = planets[source_idx]
            tgt_p = planets[target_idx]
            
            # Simple manual override to ensure scalar dispatch fidelity
            angle, _, _, _ = wrapper.get_intercept_params((src_p[2], src_p[3]), 
                                                           {'x': tgt_p[2], 'y': tgt_p[3], 'id': tgt_p[0], 'owner': tgt_p[1], 'production': tgt_p[6], 'ships': tgt_p[4]}, 
                                                           sampled_alloc.item()/5.0 if sampled_alloc.item() < 5 else 1.0, p0_obs)
            
            final_learner_moves = [[src_p[0], angle, int(src_p[4] * (sampled_alloc.item()/5.0 if sampled_alloc.item() < 5 else 0.8))]]
            if sampled_alloc.item() == 0 or not mask_vector[target_idx]: final_learner_moves = []

            obs_list = env.step([final_learner_moves] + actions)
            done = any(state.get('status') != 'ACTIVE' for state in obs_list)
            
            reward = reward_shaper.calculate_reward(p0_obs, done, total_steps)
            
            processed['action_masks'] = action_masks_grid
            ep_obs.append(processed)
            ep_targets.append(target_idx_val)
            ep_allocs.append(sampled_alloc.item())
            ep_logp.append(log_prob)
            ep_values.append(value_t.item())
            ep_rewards.append(reward)
            ep_dones.append(done)

            steps_counter += 1
            total_steps += 1

        # Bootstrap final value
        last_obs = obs_list[0]['observation']
        last_processed = obs_proc.process(last_obs, player_id=config["player_id"])
        with torch.no_grad():
            v_ent = torch.tensor(last_processed['entities'], dtype=torch.float32).unsqueeze(0).to(device)
            v_ids = torch.tensor(last_processed['entity_ids'], dtype=torch.long).unsqueeze(0).to(device)
            v_msk = torch.tensor(last_processed['mask'], dtype=torch.float32).unsqueeze(0).to(device)
            _, _, last_val = model(v_ent, v_ids, v_msk)
            ep_values.append(last_val.item())

        obs_buffer.extend(ep_obs)
        targets_buffer.extend(ep_targets)
        allocs_buffer.extend(ep_allocs)
        log_probs_buffer.extend(ep_logp)
        values_buffer.extend(ep_values)
        rewards_buffer.extend(ep_rewards)
        dones_buffer.extend(ep_dones)

        if len(rewards_buffer) >= args.batch_size:
            advantages, returns = compute_gae(np.array(rewards_buffer), np.array(values_buffer), np.array(dones_buffer), config["gamma"], config["gae_lambda"])
            rollout_data = {'obs': np.array(obs_buffer), 'targets': targets_buffer, 'allocs': allocs_buffer, 'log_probs': log_probs_buffer, 'returns': returns, 'advantages': advantages}
            model.train()
            ppo_update(model, optimizer, rollout_data, config, epochs=config["n_epochs"], batch_size=args.batch_size)
            torch.save(model.state_dict(), f'checkpoints/ppo_step_{total_steps}.pt')
            obs_buffer, targets_buffer, allocs_buffer, log_probs_buffer, values_buffer, rewards_buffer, dones_buffer = [], [], [], [], [], [], []

    print("Training complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--bc_checkpoint', type=str, default='checkpoints/bc_pretrained.pt')
    args = parser.parse_args()
    train(args)
