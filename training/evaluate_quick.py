import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kaggle_environments import make
import torch

from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper

from orbit_wars_ai.agents.transformer_ppo.policy import TransformerPPOPolicy
from orbit_wars_ai.agents.transformer_ppo.config import TransformerPPOConfig


def evaluate_quick(agent_path: str = None, num_games: int = 10, max_steps: int = 500, device: str = 'cpu'):
    env = make('orbit_wars', debug=False)
    wrapper = OrbitWarsWrapper({})
    obs_proc = ObservationProcessor()
    act_proc = ActionProcessor(wrapper)

    policy = None
    if agent_path:
        try:
            cfg = TransformerPPOConfig()
            policy = TransformerPPOPolicy(cfg, device=device)
            policy.model.load_state_dict(torch.load(agent_path, map_location=device))
            policy.model.eval()
        except Exception:
            policy = None

    win_count = 0
    print(f"Quick-evaluating {agent_path} for {num_games} games (max_steps={max_steps})")

    for g in range(num_games):
        obs_list = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            actions = []
            # for each player, build action
            for pid in range(len(obs_list)):
                current_obs = obs_list[pid]['observation']
                if pid == 0:
                    # learner
                    if policy is not None:
                        try:
                            processed = obs_proc.process(current_obs, 0)
                            tgt, alloc, _, _ = policy.get_action(processed)
                            a = act_proc.process_actions(current_obs, 0, [tgt], [alloc])
                            actions.append(a)
                        except Exception:
                            actions.append(HeuristicBaseline(0).act(current_obs))
                    else:
                        actions.append(HeuristicBaseline(0).act(current_obs))
                else:
                    actions.append(HeuristicBaseline(pid).act(current_obs))

            obs_list = env.step(actions)
            done = any(state.get('status') != 'ACTIVE' for state in obs_list)
            steps += 1

        # final reward for player 0
        try:
            reward = obs_list[0].get('reward', 0)
        except Exception:
            reward = 0
        if reward is not None and reward > 0:
            win_count += 1

    win_rate = win_count / num_games
    print(f"Quick win rate: {win_rate*100:.2f}% ({win_count}/{num_games})")
    return win_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, default=None)
    parser.add_argument('--num_games', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    rate = evaluate_quick(args.agent_path, args.num_games, args.max_steps, args.device)
    # Save summary
    os.makedirs('results', exist_ok=True)
    with open(f'results/eval_summary.txt', 'a') as f:
        f.write(f"agent={args.agent_path} num_games={args.num_games} max_steps={args.max_steps} win_rate={rate}\n")
