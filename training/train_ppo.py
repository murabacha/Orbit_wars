"""
Production-Grade Proximal Policy Optimization (PPO) Training Loop for Orbit Wars AI.
Hardware-Agile: Automatically adapts to CPU or GPU (CUDA) based on availability.
Includes Dynamic Entropy and Reward Shaping Decay.
Supports Self-Play and Baseline training.
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
from orbit_wars_ai.agents.baseline.elite_heuristic import agent as elite_heuristic_func
from online_agents.agent_1.main import agent as online_agent_1_func
from online_agents.agent_2.main import agent as online_agent_2_func
from online_agents.agent_3.main import agent as online_agent_3_func

from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.rewards import RewardShaper
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper
from training.selfplay import SelfPlayManager

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
    
    # --- THE SMART VOLUME MASK ---
    source_ships = (entities[:, :, 5] * 1000.0).view(-1)
    send_25 = source_ships * 0.25
    send_50 = source_ships * 0.50
    send_75 = source_ships * 0.75
    send_100 = source_ships * 1.0

    trickle_mask = torch.zeros((source_ships.shape[0], 6), dtype=torch.bool, device=entities.device)
    trickle_mask[:, 1] = send_25 < 15.0
    trickle_mask[:, 2] = send_50 < 15.0
    trickle_mask[:, 3] = send_75 < 15.0
    trickle_mask[:, 4] = send_100 < 15.0 # FIX: Mask 100% for tiny planets
    trickle_mask[:, 5] = True # FIX: Allocation 5 is permanently dead
    
    # THE SHUFFLE MASK: Block useless intra-empire shuffling
    target_is_owned = (entities[batch_idx, chosen_targets, 2] == 1.0)
    trickle_mask[target_is_owned, 1] = True
    trickle_mask[target_is_owned, 2] = True
    trickle_mask[target_is_owned, 3] = True

    selected_alloc_logits[trickle_mask] = -1e9
    # -----------------------------

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
        "player_id": 0, "gamma": 0.99, "gae_lambda": 0.95, "learning_rate": 1e-5,
        "clip_range": 0.05, # Max 5% policy change per update (The Straightjacket)
        "value_coef": 0.5, "entropy_coef": 0.002, "max_entities": 200,
        "n_epochs": 4, "minibatch_size": 16, "device": device
    }
    
    model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=config["max_entities"])
    checkpoint_path = args.checkpoint if args.checkpoint else args.bc_checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    wrapper_config = {"shipSpeed": 6.0, "sunRadius": 10.0, "boardSize": 100.0, "episodeSteps": 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    obs_proc = ObservationProcessor(max_entities=config["max_entities"], board_size=wrapper_config["boardSize"], max_speed=wrapper_config["shipSpeed"])
    act_proc = ActionProcessor(wrapper)
    reward_shaper = RewardShaper(player_id=config["player_id"], gamma=config["gamma"], total_training_steps=args.total_timesteps)
    
    self_play_manager = SelfPlayManager()
    if args.opponent == 'selfplay':
        self_play_manager.history.append(checkpoint_path)
        print("Self-Play Mode Activated.")

    loaded_opponents_cache = {}
    print(f"Relational Multi-Dispatch PPO initiated on: {device}")
    total_steps = args.start_step
    last_selfplay_save_step = total_steps
    episode = 0
    obs_buffer, targets_buffer, allocs_buffer = [], [], []
    log_probs_buffer, returns_buffer, advantages_buffer = [], [], []

    while total_steps < args.total_timesteps:
        episode += 1
        env = make("orbit_wars", debug=False)
        num_players = random.choice([2, 4]) if args.opponent == 'selfplay' else 2
        obs_list = env.reset(num_players)
        num_players = len(obs_list)
        
        opponent_paths = []
        opp_models = []
        
        heuristic_crucible = [
            "baseline_heuristic", 
            "elite_heuristic",
            "online_agent_1",
            "online_agent_2",
            "online_agent_3"
        ]

        if args.opponent == 'selfplay':
            for _ in range(num_players - 1):
                rand_val = random.random()
                if rand_val < 0.30:
                    opponent_paths.append(random.choice(heuristic_crucible))
                elif rand_val < 0.45:
                    opponent_paths.append(args.bc_checkpoint)
                else:
                    opponent_paths.append(self_play_manager.get_opponent())

            # Load models from cache to save VRAM
            for opp_path in opponent_paths:
                if opp_path in heuristic_crucible:
                    opp_models.append(opp_path)
                    continue
                    
                if opp_path not in loaded_opponents_cache:
                    if len(loaded_opponents_cache) >= 5: 
                        oldest_key = next(iter(loaded_opponents_cache))
                        del loaded_opponents_cache[oldest_key]
                        if device_type == 'cuda': torch.cuda.empty_cache()
                    loaded_opponents_cache[opp_path] = torch.load(opp_path, map_location=device)
                
                temp_model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=200).to(device)
                temp_model.load_state_dict(loaded_opponents_cache[opp_path])
                temp_model.eval()
                opp_models.append(temp_model)
            
            print(f"\n⚔️  Starting {num_players}-Player Match (Opponents: {[p.split('/')[-1] if '/' in p else p for p in opponent_paths]})")
        else:
            opp_models = ["baseline_heuristic"] * (num_players - 1)
            opponent_paths = ["baseline_heuristic"] * (num_players - 1)
        
        ep_obs, ep_targets, ep_allocs, ep_logp, ep_values, ep_rewards, ep_dones = [], [], [], [], [], [], []
        total_ep_reward = 0
        done = False
        steps_counter = 0
        my_slot = 0
        baselines = [HeuristicBaseline(pid) for pid in range(num_players)]
        
        while not done and steps_counter < 500:
            actions = [[] for _ in range(num_players)]
            for pid in range(num_players):
                if obs_list[pid]['status'] != 'ACTIVE': continue
                raw_obs = obs_list[pid]['observation']
                if pid == my_slot:
                    processed = obs_proc.process(raw_obs, player_id=pid)
                    full_mask = wrapper.get_action_mask(raw_obs, player_id=pid)
                    action_masks_grid = np.zeros((config["max_entities"], config["max_entities"]), dtype=bool)
                    action_masks_grid[:full_mask.shape[0], :full_mask.shape[1]] = full_mask
                    
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
                            
                            # --- THE SMART VOLUME MASK ---
                            chosen_targets = sampled_targets.view(-1)
                            source_ships = (ent_t[:, :, 5] * 1000.0).view(-1)
                            send_25 = source_ships * 0.25
                            send_50 = source_ships * 0.50
                            send_75 = source_ships * 0.75
                            send_100 = source_ships * 1.0
                            
                            trickle_mask = torch.zeros((source_ships.shape[0], 6), dtype=torch.bool, device=device)
                            trickle_mask[:, 1] = send_25 < 15.0
                            trickle_mask[:, 2] = send_50 < 15.0
                            trickle_mask[:, 3] = send_75 < 15.0
                            trickle_mask[:, 4] = send_100 < 15.0
                            trickle_mask[:, 5] = True
                            
                            target_is_owned = (ent_t[batch_idx, chosen_targets, 2] == 1.0)
                            trickle_mask[target_is_owned, 1] = True
                            trickle_mask[target_is_owned, 2] = True
                            trickle_mask[target_is_owned, 3] = True
                            
                            selected_alloc_logits[trickle_mask] = -1e9
                            # -----------------------------

                            alloc_dist = torch.distributions.Categorical(logits=selected_alloc_logits); sampled_allocs = alloc_dist.sample().view(B, N)
                            is_source_owned = (processed['entities'][:, 2] == 1.0); valid_source_mask = is_source_owned & (processed['mask'] == 1.0)
                            log_p_target = target_dist.log_prob(sampled_targets); log_p_alloc = alloc_dist.log_prob(sampled_allocs.view(-1)).view(B, N)
                            joint_log_prob = (((log_p_target + log_p_alloc) * torch.tensor(valid_source_mask, device=device).float()).sum(dim=-1)).item()
                    
                    actions[pid] = act_proc.process_actions(raw_obs, player_id=pid, target_indices=sampled_targets.squeeze(0).cpu().numpy().tolist(), allocation_indices=sampled_allocs.squeeze(0).cpu().numpy().tolist())
                    processed['action_masks'] = action_masks_grid; ep_obs.append(processed); ep_targets.append(sampled_targets.squeeze(0).cpu().numpy()); ep_allocs.append(sampled_allocs.squeeze(0).cpu().numpy()); ep_logp.append(joint_log_prob); ep_values.append(value_t.item()); ep_dones.append(False)
                else:
                    opp_idx = pid - 1
                    opp_m = opp_models[opp_idx]
                    
                    if opp_m in heuristic_crucible:
                        if opp_m == "baseline_heuristic":
                            actions[pid] = baselines[pid].act(raw_obs)
                        elif opp_m == "elite_heuristic":
                            # Call it exactly like the online agents
                            actions[pid] = elite_heuristic_func(raw_obs, wrapper_config)
                        elif opp_m == "online_agent_1":
                            actions[pid] = online_agent_1_func(raw_obs)
                        elif opp_m == "online_agent_2":
                            actions[pid] = online_agent_2_func(raw_obs)
                        elif opp_m == "online_agent_3":
                            actions[pid] = online_agent_3_func(raw_obs)
                    else:
                        p_obs = obs_proc.process(raw_obs, player_id=pid)
                        f_mask = wrapper.get_action_mask(raw_obs, player_id=pid); a_masks = np.zeros((config["max_entities"], config["max_entities"]), dtype=bool); a_masks[:f_mask.shape[0], :f_mask.shape[1]] = f_mask
                        with torch.no_grad():
                            ent_o = torch.tensor(p_obs['entities'], dtype=torch.float32).unsqueeze(0).to(device); ids_o = torch.tensor(p_obs['entity_ids'], dtype=torch.long).unsqueeze(0).to(device); msk_o = torch.tensor(p_obs['mask'], dtype=torch.float32).unsqueeze(0).to(device); amsk_o = torch.tensor(a_masks, dtype=torch.bool).unsqueeze(0).to(device)
                            t_logits, a_logits, _ = opp_m(ent_o, ids_o, msk_o, amsk_o)
                            
                            source_idx_o = torch.arange(config["max_entities"], device=device)
                            t_acts = t_logits.squeeze(0).argmax(dim=-1)
                            selected_alloc_logits_o = a_logits.squeeze(0)[source_idx_o, t_acts, :]
                            
                            # --- THE SMART VOLUME MASK ---
                            source_ships_o = (ent_o[:, :, 5] * 1000.0).view(-1)
                            send_25_o = source_ships_o * 0.25
                            send_50_o = source_ships_o * 0.50
                            send_75_o = source_ships_o * 0.75
                            send_100_o = source_ships_o * 1.0
                            
                            trickle_mask_o = torch.zeros((source_ships_o.shape[0], 6), dtype=torch.bool, device=device)
                            trickle_mask_o[:, 1] = send_25_o < 15.0
                            trickle_mask_o[:, 2] = send_50_o < 15.0
                            trickle_mask_o[:, 3] = send_75_o < 15.0
                            trickle_mask_o[:, 4] = send_100_o < 15.0
                            trickle_mask_o[:, 5] = True
                            
                            target_is_owned_o = (ent_o[0, t_acts, 2] == 1.0)
                            trickle_mask_o[target_is_owned_o, 1] = True
                            trickle_mask_o[target_is_owned_o, 2] = True
                            trickle_mask_o[target_is_owned_o, 3] = True
                            
                            selected_alloc_logits_o[trickle_mask_o] = -1e9
                            # -----------------------------
                            
                            a_acts = selected_alloc_logits_o.argmax(dim=-1).cpu().numpy().tolist()
                            t_acts = t_acts.cpu().numpy().tolist()
                        actions[pid] = act_proc.process_actions(raw_obs, player_id=pid, target_indices=t_acts, allocation_indices=a_acts)
            
            obs_list = env.step(actions)
            real_done = (obs_list[my_slot].get('status') != 'ACTIVE') or all(s['status'] != 'ACTIVE' for i, s in enumerate(obs_list) if i != my_slot)
            done = real_done
            
            reward = reward_shaper.calculate_reward(obs_list[my_slot]['observation'], real_done, total_steps, steps_counter)
            ep_rewards.append(reward); total_ep_reward += reward; steps_counter += 1; total_steps += 1
            
        if len(ep_dones) > 0: ep_dones[-1] = real_done
        if args.opponent == 'selfplay' and len(opponent_paths) > 0:
            final_reward = obs_list[my_slot].get('reward', 0)
            self_play_manager.update_win_rate(opponent_paths[0], won=(final_reward > 0))

        last_obs = obs_list[my_slot]['observation']; last_processed = obs_proc.process(last_obs, player_id=my_slot)
        with torch.no_grad():
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                v_ent = torch.tensor(last_processed['entities'], dtype=torch.float32).unsqueeze(0).to(device); v_ids = torch.tensor(last_processed['entity_ids'], dtype=torch.long).unsqueeze(0).to(device); v_msk = torch.tensor(last_processed['mask'], dtype=torch.float32).unsqueeze(0).to(device); _, _, last_val = model(v_ent, v_ids, v_msk)
                bootstrap_val = 0.0 if real_done else last_val.item()
                
        ep_adv, ep_ret = compute_gae(np.array(ep_rewards), np.array(ep_values), np.array(ep_dones), config["gamma"], config["gae_lambda"], bootstrap_val)
        obs_buffer.extend(ep_obs); targets_buffer.extend(ep_targets); allocs_buffer.extend(ep_allocs); log_probs_buffer.extend(ep_logp); advantages_buffer.extend(ep_adv); returns_buffer.extend(ep_ret)
        
        opp_names = [os.path.basename(p) if '/' in p else p for p in opponent_paths]
        print(f"E{episode} | Opps: {opp_names} | Steps: {steps_counter} | Reward: {total_ep_reward:.2f} | Buffer: {len(returns_buffer)}/{args.batch_size}")
        
        if len(returns_buffer) >= args.batch_size:
            # FINE-TUNING DECAY: 10x smaller to protect BC weights
            lr_start = 1e-5
            lr_end = 1e-6
            decay_fraction = min(1.0, total_steps / args.total_timesteps)
            config["learning_rate"] = lr_start - decay_fraction * (lr_start - lr_end)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config["learning_rate"]

            # FINE-TUNING ENTROPY: Stop it from hallucinating random moves
            ent_start = 0.002
            ent_end = 0.0001
            config["entropy_coef"] = ent_start - decay_fraction * (ent_start - ent_end)

            rollout_data = {'obs': np.array(obs_buffer), 'targets': targets_buffer, 'allocs': allocs_buffer, 'log_probs': log_probs_buffer, 'returns': returns_buffer, 'advantages': advantages_buffer}
            model.train(); up_metrics = ppo_update(model, optimizer, rollout_data, config, epochs=config["n_epochs"], minibatch_size=config["minibatch_size"])
            print(f"\n--- PPO Update @ Step {total_steps} ---")
            print(f"Policy Loss: {up_metrics['pg_loss']:.4f} | Value Loss: {up_metrics['v_loss']:.4f} | Entropy: {up_metrics['entropy']:.4f}")
            print(f"Learning Rate: {config['learning_rate']:.7f} | Entropy Coef: {config['entropy_coef']:.4f}")
            print("------------------------------------\n")
            
            os.makedirs("checkpoints", exist_ok=True)
            local_save_path = f'checkpoints/ppo_step_{total_steps}.pt'
            torch.save(model.state_dict(), local_save_path)
            
            if args.opponent == 'selfplay' and (total_steps - last_selfplay_save_step >= 20000):
                self_play_manager.save_checkpoint(model, total_steps)
                last_selfplay_save_step = total_steps

            gdrive_path = '/content/drive/MyDrive/OrbitWars_Checkpoints'
            if os.path.exists(gdrive_path):
                import shutil
                try: shutil.copy2(local_save_path, os.path.join(gdrive_path, f'ppo_step_{total_steps}.pt'))
                except: pass
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
    parser.add_argument('--opponent', type=str, default='baseline', choices=['baseline', 'selfplay'])
    args = parser.parse_args()
    train(args)
