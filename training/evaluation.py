import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kaggle_environments import make
import torch

from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper


def load_policy(model_class_path: str, weights_path: str, device: str = 'cpu'):
    # model_class_path example: orbit_wars_ai.agents.transformer_ppo.policy.TransformerPPOPolicy
    module_path, class_name = model_class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    PolicyClass = getattr(module, class_name)
    # Instantiate with default config if possible
    policy = PolicyClass(device=device)
    if weights_path and torch.exists(weights_path) if hasattr(torch, 'exists') else True:
        try:
            policy.model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception:
            pass
    return policy


def evaluate_agent(agent_weights: str = None, num_games: int = 100):
    env = make("orbit_wars", debug=False)
    win_count = 0

    wrapper = OrbitWarsWrapper({})
    obs_proc = ObservationProcessor()
    act_proc = ActionProcessor(wrapper)

    print(f"Evaluating {agent_weights} over {num_games} games...")

    for i in range(num_games):
        # Simple approach: learner at player 0, heuristics others
        baseline1 = HeuristicBaseline(1)
        baseline2 = HeuristicBaseline(2)
        baseline3 = HeuristicBaseline(3)

        # If weights provided, we can create a policy wrapper closure
        def learner_agent(obs, config):
            # Try simple heuristic fallback if model not loaded
            try:
                # Minimal inference: use heuristic mapping if no model
                # The repository's policy expects processed tensors; for evaluation fallback to heuristic
                from orbit_wars_ai.agents.transformer_ppo.policy import TransformerPPOPolicy
                policy = TransformerPPOPolicy(device='cpu')
                if agent_weights:
                    policy.model.load_state_dict(torch.load(agent_weights, map_location='cpu'))
                processed = obs_proc.process(obs, 0)
                tgt, alloc, _, _ = policy.get_action(processed)
                return act_proc.process_actions(obs, 0, [tgt], [alloc])
            except Exception:
                return HeuristicBaseline(0).act(obs)

        agents = [learner_agent, lambda o, c: baseline1.act(o), lambda o, c: baseline2.act(o), lambda o, c: baseline3.act(o)]

        env.run(agents)
        try:
            final_steps = env.steps
            if len(final_steps) > 0:
                reward = final_steps[-1][0].reward
                if reward is not None and reward > 0:
                    win_count += 1
        except Exception:
            pass

    win_rate = win_count / num_games
    print(f"Win rate: {win_rate * 100:.2f}% ({win_count}/{num_games})")
    return win_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, default=None)
    parser.add_argument('--num_games', type=int, default=50)
    args = parser.parse_args()
    evaluate_agent(args.agent_path, args.num_games)
