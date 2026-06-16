"""
Generates a visual HTML replay of the PPO agent
playing against the Elite Heuristic.
"""
import os
import sys
import torch
import numpy as np
from kaggle_environments import make

# Enforce clean path insertions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.evaluation import EvaluationTournament
from orbit_wars_ai.agents.baseline.elite_heuristic import agent as elite_agent

def render_ppo_vs_elite(agent_path, output_path="ppo_vs_elite.html"):
    # 1. Initialize the evaluation logic
    eval_suite = EvaluationTournament(agent_path=agent_path, device='cpu')
    
    # 2. Setup environment (2-player match)
    env = make("orbit_wars", debug=True)
    obs_list = env.reset(2)
    
    num_players = len(obs_list)
    my_slot = 0
    elite_slot = 1
    
    print(f"🎬 Starting match. PPO Agent (Player {my_slot}, Blue) vs Elite Heuristic (Player {elite_slot}, Red)")
    
    done = False
    step = 0
    while not done and step < 500:
        actions = [[] for _ in range(num_players)]
        for pid in range(num_players):
            if obs_list[pid]['status'] != 'ACTIVE':
                continue
                
            raw_obs = obs_list[pid]['observation']
            if pid == my_slot:
                p_obs = eval_suite.obs_proc.process(raw_obs, player_id=my_slot)
                actions[my_slot] = eval_suite.get_agent_actions(p_obs, raw_obs, player_id=my_slot)
            else:
                actions[pid] = elite_agent(raw_obs)
                
        obs_list = env.step(actions)
        # Match ends if anyone is not active (or 500 steps)
        done = any(state.get('status') != 'ACTIVE' for state in obs_list)
        step += 1
        if step % 50 == 0:
            print(f"Step {step}...")

    # 3. Save the HTML replay
    with open(output_path, "w") as f:
        f.write(env.render(mode="html", width=800, height=600))
    
    print(f"✅ Replay saved to: {output_path}")
    print(f"🏁 Final Outcome: {[s.get('reward', 0) for s in obs_list]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, default="Orbit_wars/checkpoints/ppo_step_199030.pt")
    parser.add_argument('--output_path', type=str, default="ppo_vs_elite_199030.html")
    args = parser.parse_args()
    
    if not os.path.exists(args.agent_path):
        # Check relative path
        alt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", os.path.basename(args.agent_path))
        if os.path.exists(alt_path):
            args.agent_path = alt_path
        
    render_ppo_vs_elite(args.agent_path, output_path=args.output_path)
