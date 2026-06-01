"""
Generates a visual HTML replay of the Relational Transformer agent 
playing against 3 fixed heuristic baselines.
"""
import os
import sys
import torch
import numpy as np
from kaggle_environments import make

# Enforce clean path insertions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.evaluation import EvaluationTournament

def render_sample_match(agent_path, output_path="match_replay.html"):
    # 1. Initialize the evaluation logic (handles model loading and processing)
    # We use CPU since this is just inference
    eval_suite = EvaluationTournament(agent_path=agent_path, device='cpu')
    
    # 2. Setup environment
    env = make("orbit_wars", debug=True)
    obs_list = env.reset()
    num_players = len(obs_list)
    
    # We'll put our agent in Slot 0 (Blue) for easiest tracking
    my_slot = 0
    colors = ["BLUE", "RED", "GREEN", "YELLOW"]
    print(f"🎬 Starting match. Our agent is Player {my_slot} ({colors[my_slot]})")
    
    from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline
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
                p_obs = eval_suite.obs_proc.process(raw_obs, player_id=my_slot)
                actions[my_slot] = eval_suite.get_agent_actions(p_obs, raw_obs, player_id=my_slot)
            else:
                actions[pid] = bots[pid].act(raw_obs)
                
        obs_list = env.step(actions)
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
    parser.add_argument('--agent_path', type=str, required=True)
    args = parser.parse_args()
    
    render_sample_match(args.agent_path)
