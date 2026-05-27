import argparse
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from kaggle_environments import make

from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.rewards import RewardShaper
from orbit_wars_ai.agents.transformer_ppo.policy import TransformerPPOPolicy
from orbit_wars_ai.agents.transformer_ppo.config import TransformerPPOConfig
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline


def compute_gae(rewards, values, masks, gamma, lam):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - masks[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_update(policy, optimizer, obs_batch, action_targets, action_allocs, old_log_probs, returns, advantages, clip_range, value_coef, entropy_coef, epochs=4, batch_size=64):
    dataset_size = len(obs_batch)
    inds = np.arange(dataset_size)
    for _ in range(epochs):
        np.random.shuffle(inds)
        for start in range(0, dataset_size, batch_size):
            mb_inds = inds[start:start + batch_size]

            # Prepare tensors
            entities = torch.tensor(np.stack([o['entities'] for o in obs_batch])[mb_inds]).float().to(policy.device)
            entity_ids = torch.tensor(np.stack([o['entity_ids'] for o in obs_batch])[mb_inds]).long().to(policy.device)
            mask = torch.tensor(np.stack([o['mask'] for o in obs_batch])[mb_inds]).float().to(policy.device)

            target_actions = torch.tensor(action_targets[mb_inds]).to(policy.device)
            alloc_actions = torch.tensor(action_allocs[mb_inds]).to(policy.device)
            old_logp = torch.tensor(old_log_probs[mb_inds]).float().to(policy.device)
            returns_t = torch.tensor(returns[mb_inds]).float().to(policy.device)
            adv_t = torch.tensor(advantages[mb_inds]).float().to(policy.device)

            # Forward
            target_logits, alloc_logits, values = policy.model(entities, entity_ids, mask)

            # We assume actions correspond to first valid token for simplicity
            # Convert logits shape: target_logits [batch, seq_len]
            # Pick the logits at the indices specified by target_actions
            batch_idx = torch.arange(len(mb_inds)).to(policy.device)
            chosen_target_logits = target_logits[batch_idx, target_actions[mb_inds]]
            target_dist = torch.distributions.Categorical(logits=target_logits)
            alloc_dist = torch.distributions.Categorical(logits=alloc_logits)

            logp = target_dist.log_prob(target_actions[mb_inds]) + alloc_dist.log_prob(alloc_actions[mb_inds])
            entropy = target_dist.entropy().mean() + alloc_dist.entropy().mean()

            # Policy loss
            ratio = torch.exp(logp - old_logp[mb_inds])
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns_t)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.5)
            optimizer.step()


def train(args):
    config = TransformerPPOConfig()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    policy = TransformerPPOPolicy(config, device=device)
    optimizer = optim.Adam(policy.model.parameters(), lr=config.learning_rate)

    wrapper = OrbitWarsWrapper({})
    obs_proc = ObservationProcessor(wrapper, max_entities=config.max_entities)
    act_proc = ActionProcessor(wrapper)
    reward_shaper = RewardShaper(config.player_id)

    print(f"Starting PPO training on {device}...")

    total_steps = 0
    episode = 0

    # Training buffers
    obs_buffer = []
    target_actions = []
    alloc_actions = []
    log_probs = []
    values = []
    rewards = []
    masks = []

    # Opponent factory
    def make_baseline(player_id):
        baseline = HeuristicBaseline(player_id)
        def agent(obs, config):
            return baseline.act(obs)
        return agent

    while total_steps < args.total_timesteps:
        episode += 1
        env = make("orbit_wars", debug=False)

        # Per-episode buffers
        ep_obs = []
        ep_targets = []
        ep_allocs = []
        ep_logp = []
        ep_values = []
        ep_rewards = []
        ep_masks = []

        # We'll collect data by registering an agent function that appends to local buffers
        def learner_agent(obs, config):
            # obs is the Kaggle observation dict
            processed = obs_proc.process(obs, config.get('player_id', 0))
            tgt, alloc, logp, val = policy.get_action(processed, action_mask=None)

            # Store tensors/arrays for training
            ep_obs.append(processed)
            ep_targets.append(tgt)
            ep_allocs.append(alloc)
            ep_logp.append(logp)
            ep_values.append(val)

            # Return moves
            moves = act_proc.process_actions(obs, 0, [tgt], [alloc])
            return moves

        # Build agents list: learner at player 0, heuristics others
        agents = [learner_agent]
        for pid in range(1, 4):
            agents.append(make_baseline(pid))

        # Run one full episode
        steps = env.run(agents)

        # Kaggle env does not return stepwise rewards directly here; instead we use final rewards
        # For simplicity, assign zero intermediate rewards and final reward at the end
        final_rewards = None
        try:
            final_steps = env.steps
            if len(final_steps) > 0:
                final_rewards = final_steps[-1][0].reward
        except Exception:
            final_rewards = None

        T = len(ep_values)
        for i in range(T):
            ep_rewards.append(0.0)
            ep_masks.append(0.0)

        # If final reward exists, add to last step
        if final_rewards is not None and len(ep_rewards) > 0:
            try:
                ep_rewards[-1] += float(final_rewards)
            except Exception:
                pass

        # Convert lists to arrays and extend global buffers
        obs_buffer.extend(ep_obs)
        target_actions.extend(ep_targets)
        alloc_actions.extend(ep_allocs)
        log_probs.extend(ep_logp)
        values.extend(ep_values)
        rewards.extend(ep_rewards)
        masks.extend([0.0] * len(ep_rewards))

        total_steps = len(rewards)
        print(f"Episode {episode} collected {len(ep_rewards)} steps, total steps {total_steps}")

        # When enough steps collected, perform update
        if total_steps >= args.batch_size:
            # Prepare values array with an extra bootstrap value of 0
            vals = np.array(values + [0.0], dtype=np.float32)
            rews = np.array(rewards, dtype=np.float32)
            msks = np.array(masks + [0.0], dtype=np.float32)

            advantages, returns = compute_gae(rews, vals, msks, config.gamma, config.gae_lambda)

            # Convert buffers to numpy arrays for ppo_update
            ppo_update(policy, optimizer, np.array(obs_buffer), np.array(target_actions), np.array(alloc_actions), np.array(log_probs), returns, advantages, config.clip_range, config.value_coef, config.entropy_coef, epochs=config.n_epochs, batch_size=args.batch_size)

            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = f'checkpoints/ppo_step_{total_steps}.pt'
            torch.save(policy.model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            # Clear buffers
            obs_buffer = []
            target_actions = []
            alloc_actions = []
            log_probs = []
            values = []
            rewards = []
            masks = []

    print("Training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total environment steps to run')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size / rollout length before update')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=50000)
    args = parser.parse_args()
    train(args)
