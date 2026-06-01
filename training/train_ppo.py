"""
Production-Grade Proximal Policy Optimization (PPO) Training Loop for Orbit Wars AI.
Hardware-Agile: Automatically adapts to CPU or GPU (CUDA) based on availability.
Includes Dynamic Entropy and Reward Shaping Decay.
"""
import argparse
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kaggle_environments import make

from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.rewards import RewardShaper
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper

def compute_gae(rewards, values, dones, gamma, gae_lambda, last_value):
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    padded_values = np.append(values, last_value)
    for t in reversed(range(len(rewards))):
        next_val = padded_values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lam
    returns = advantages + values
    return advantages, returns

def get_joint_log_prob(model, entities, entity_ids, mask, target_actions, alloc_actions, action_masks=None):
    target_logits, alloc_logits, values = model(entities, entity_ids, mask, action_masks)
    B, N, _ = entities.shape
    target_dist = torch.distributions.Categorical(logits=target_logits)
    batch_idx = torch.arange(B, device=entities.device).unsqueeze(1).expand(-1, N).reshape(-1)
    source_idx = torch.arange(N, device=entities.device).unsqueeze(0).expand(B, -1).reshape(-1)
    chosen_targets = target_actions.view(-1)
    selected_alloc_logits = alloc_logits[batch_idx, source_idx, chosen_targets, :]
    alloc_dist = torch.distributions.Categorical(logits=selected_alloc_logits)
    is_source_owned = (entities[:, :, 2] == 1.0)
    valid_source_mask = is_source_owned & (mask == 1.0)
    log_p_target = target_dist.log_prob(target_actions)
    log_p_alloc = alloc_dist.log_prob(alloc_actions.view(-1)).view(B, N)
    joint_log_prob = ((log_p_target + log_p_alloc) * valid_source_mask.float()).sum(dim=-1)
    total_entropy = ((target_dist.entropy() + alloc_dist.entropy().view(B, N)) * valid_source_mask.float()).sum(dim=-1) / torch.clamp(valid_source_mask.sum(dim=-1), min=1.0)
    return joint_log_prob, total_entropy.mean(), values.squeeze(-1)

def ppo_update(model: nn.Module, optimizer: optim.Optimizer, rollout_data: dict, 
               config: dict, epochs: int = 4, minibatch_size: int = 32):
    obs_batch = rollout_data['obs']
    device = config["device"]
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    action_targets = torch.tensor(np.stack(rollout_data['targets']), dtype=torch.long, device=device)
    action_allocs = torch.tensor(np.stack(rollout_data['allocs']), dtype=torch.long, device=device)
    old_log_probs = torch.tensor(np.array(rollout_data['log_probs']), dtype=torch.float32, device=device)
    returns = torch.tensor(np.array(rollout_data['returns']), dtype=torch.float32, device=device)
    advantages = torch.tensor(np.array(rollout_data['advantages']), dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
    dataset_size = len(obs_batch)
    inds = np.arange(dataset_size)
    metrics = {"pg_loss": [], "v_loss": [], "entropy": []}
    for epoch in range(epochs):
        np.random.shuffle(inds)
        for start in range(0, dataset_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = inds[start:end]
            mb_obs = obs_batch[mb_inds]
            entities = torch.tensor(np.stack([o['entities'] for o in mb_obs]), dtype=torch.float32, device=device)
            ids = torch.tensor(np.stack([o['entity_ids'] for o in mb_obs]), dtype=torch.long, device=device)
            mask = torch.tensor(np.stack([o['mask'] for o in mb_obs]), dtype=torch.float32, device=device)
            amasks = torch.tensor(np.stack([o['action_masks'] for o in mb_obs]), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                new_log_probs, entropy, values = get_joint_log_prob(model, entities, ids, mask, action_targets[mb_inds], action_allocs[mb_inds], amasks)
                ratio = torch.exp(new_log_probs - old_log_probs[mb_inds])
                surr1 = ratio * advantages[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - config["clip_range"], 1.0 + config["clip_range"]) * advantages[mb_inds]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[mb_inds])
                loss = policy_loss + config["value_coef"] * value_loss - config["entropy_coef"] * entropy
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            metrics["pg_loss"].append(policy_loss.item()); metrics["v_loss"].append(value_loss.item()); metrics["entropy"].append(entropy.item())
    return {k: np.mean(v) for k, v in metrics.items()}

def train(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    config = {
        "player_id": 0, "gamma": 0.99, "gae_lambda": 0.95, "learning_rate": 2.5e-4,
        "clip_range": 0.2, "value_coef": 0.5, "entropy_coef": 0.01, "max_entities": 200,
        "n_epochs": 4, "minibatch_size": 16, "device": device
    }
    
    model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=config["max_entities"])
    checkpoint_path = args.checkpoint if args.checkpoint else args.bc_checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
    wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    obs_proc = ObservationProcessor(max_entities=config["max_entities"], board_size=wrapper_config["boardSize"], max_speed=wrapper_config["shipSpeed"])
    act_proc = ActionProcessor(wrapper)
    reward_shaper = RewardShaper(player_id=config["player_id"], gamma=config["gamma"], total_training_steps=args.total_timesteps)

    print(f"Relational Multi-Dispatch PPO initiated on: {device}")
    total_steps = args.start_step
    episode = 0
    obs_buffer, targets_buffer, allocs_buffer = [], [], []
    log_probs_buffer, returns_buffer, advantages_buffer = [], [], []

    while total_steps < args.total_timesteps:
        episode += 1
        env = make("orbit_wars", debug=False)
        obs_list = env.reset()
        num_players = len(obs_list)
        ep_obs, ep_targets, ep_allocs, ep_logp, ep_values, ep_rewards, ep_dones = [], [], [], [], [], [], []
        total_ep_reward = 0
        done = False
        steps_counter = 0
        baselines = [HeuristicBaseline(pid) for pid in range(num_players)]
        while not done and steps_counter < 500:
            actions = []
            for pid in range(1, num_players):
                if obs_list[pid]['status'] == 'ACTIVE':
                    actions.append(baselines[pid].act(obs_list[pid]['observation']))
                else: actions.append([])
            p0_obs = obs_list[0]['observation']
            processed = obs_proc.process(p0_obs, player_id=config["player_id"])
            mask_vector = wrapper.get_action_mask(p0_obs, player_id=config["player_id"], allocation_percentage=1.0)
            action_masks_grid = np.zeros((config["max_entities"], config["max_entities"]), dtype=bool)
            planets_raw = p0_obs.get("planets", [])
            for s_idx in range(len(planets_raw)):
                if planets_raw[s_idx][1] == config["player_id"]:
                    action_masks_grid[s_idx, :len(planets_raw)] = mask_vector
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                    ent_t = torch.tensor(processed['entities'], dtype=torch.float32).unsqueeze(0).to(device)
                    ids_t = torch.tensor(processed['entity_ids'], dtype=torch.long).unsqueeze(0).to(device)
                    msk_t = torch.tensor(processed['mask'], dtype=torch.float32).unsqueeze(0).to(device)
                    amsk_t = torch.tensor(action_masks_grid, dtype=torch.bool).unsqueeze(0).to(device)
                    target_logits, alloc_logits, value_t = model(ent_t, ids_t, msk_t, amsk_t)
                    target_dist = torch.distributions.Categorical(logits=target_logits); sampled_targets = target_dist.sample()
                    B, N, _ = ent_t.shape
                    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1); source_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
                    selected_alloc_logits = alloc_logits[batch_idx, source_idx, sampled_targets.view(-1), :]
                    alloc_dist = torch.distributions.Categorical(logits=selected_alloc_logits); sampled_allocs = alloc_dist.sample().view(B, N)
                    is_source_owned = (processed['entities'][:, 2] == 1.0); valid_source_mask = is_source_owned & (processed['mask'] == 1.0)
                    log_p_target = target_dist.log_prob(sampled_targets); log_p_alloc = alloc_dist.log_prob(sampled_allocs.view(-1)).view(B, N)
                    joint_log_prob = ((log_p_target + log_p_alloc) * torch.tensor(valid_source_mask, device=device).float()).sum(dim=-1).item()
            learner_moves = act_proc.process_actions(p0_obs, player_id=config["player_id"], target_indices=sampled_targets.squeeze(0).cpu().numpy()[np.where(valid_source_mask)[0]].tolist(), allocation_indices=sampled_allocs.squeeze(0).cpu().numpy()[np.where(valid_source_mask)[0]].tolist())
            obs_list = env.step([learner_moves] + actions); done = (obs_list[0].get('status') != 'ACTIVE')
            reward = reward_shaper.calculate_reward(p0_obs, done, total_steps); total_ep_reward += reward
            processed['action_masks'] = action_masks_grid; ep_obs.append(processed); ep_targets.append(sampled_targets.squeeze(0).cpu().numpy()); ep_allocs.append(sampled_allocs.squeeze(0).cpu().numpy()); ep_logp.append(joint_log_prob); ep_values.append(value_t.item()); ep_rewards.append(reward); ep_dones.append(done)
            steps_counter += 1; total_steps += 1
        last_obs = obs_list[0]['observation']; last_processed = obs_proc.process(last_obs, player_id=config["player_id"])
        with torch.no_grad():
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                v_ent = torch.tensor(last_processed['entities'], dtype=torch.float32).unsqueeze(0).to(device); v_ids = torch.tensor(last_processed['entity_ids'], dtype=torch.long).unsqueeze(0).to(device); v_msk = torch.tensor(last_processed['mask'], dtype=torch.float32).unsqueeze(0).to(device); _, _, last_val = model(v_ent, v_ids, v_msk); bootstrap_val = last_val.item()
        ep_adv, ep_ret = compute_gae(np.array(ep_rewards), np.array(ep_values), np.array(ep_dones), config["gamma"], config["gae_lambda"], bootstrap_val)
        obs_buffer.extend(ep_obs); targets_buffer.extend(ep_targets); allocs_buffer.extend(ep_allocs); log_probs_buffer.extend(ep_logp); advantages_buffer.extend(ep_adv); returns_buffer.extend(ep_ret)
        print(f"E{episode} | Steps: {steps_counter} | Reward: {total_ep_reward:.2f} | Buffer: {len(returns_buffer)}/{args.batch_size}")
        if len(returns_buffer) >= args.batch_size:
            # DYNAMIC ENTROPY DECAY: Linear from 0.01 to 0.001
            ent_start, ent_end = 0.01, 0.001
            decay_fraction = min(1.0, total_steps / args.total_timesteps)
            config["entropy_coef"] = ent_start - decay_fraction * (ent_start - ent_end)

            rollout_data = {'obs': np.array(obs_buffer), 'targets': targets_buffer, 'allocs': allocs_buffer, 'log_probs': log_probs_buffer, 'returns': returns_buffer, 'advantages': advantages_buffer}
            model.train(); up_metrics = ppo_update(model, optimizer, rollout_data, config, epochs=config["n_epochs"], minibatch_size=config["minibatch_size"])
            print(f"\n--- PPO Update @ Step {total_steps} ---")
            print(f"Policy Loss: {up_metrics['pg_loss']:.4f} | Value Loss: {up_metrics['v_loss']:.4f} | Entropy: {up_metrics['entropy']:.4f}")
            print(f"Dense Weight: {reward_shaper.last_dense_weight:.3f} | Entropy Coef: {config['entropy_coef']:.4f}")
            print("------------------------------------\n")
            
            os.makedirs("checkpoints", exist_ok=True)
            local_save_path = f'checkpoints/ppo_step_{total_steps}.pt'
            torch.save(model.state_dict(), local_save_path)
            gdrive_path = '/content/drive/MyDrive/OrbitWars_Checkpoints'
            if os.path.exists(gdrive_path):
                import shutil
                try:
                    shutil.copy2(local_save_path, os.path.join(gdrive_path, f'ppo_step_{total_steps}.pt'))
                    print(f"💾 PPO Update Complete. Checkpoint saved and synced to Drive at step {total_steps}.")
                except Exception as e:
                    print(f"💾 Checkpoint saved locally, but Drive sync failed: {e}")
            else:
                print(f"💾 PPO Update Complete. Checkpoint saved locally at step {total_steps}.")
            obs_buffer, targets_buffer, allocs_buffer, log_probs_buffer, returns_buffer, advantages_buffer = [], [], [], [], [], []
    print("Training complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--bc_checkpoint', type=str, default='checkpoints/bc_pretrained.pt')
    parser.add_argument('--start_step', type=int, default=0)
    args = parser.parse_args()
    train(args)
