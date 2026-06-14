"""
Custom match renderer for comparing different agent combinations.
"""
import os
import sys
import torch
import numpy as np
from kaggle_environments import make

# Enforce clean path insertions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.evaluation import EvaluationTournament
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline

def render_match(agent_types, agent_paths, output_path="match_replay.html"):
    """
    agent_types: list of strings, either 'heuristic' or 'model'
    agent_paths: list of strings, paths to models (use None for heuristic)
    """
    num_players = len(agent_types)
    env = make("orbit_wars", debug=True)
    obs_list = env.reset(num_players)
    
    # Initialize models
    eval_suites = {}
    for i, (atype, apath) in enumerate(zip(agent_types, agent_paths)):
        if atype == 'model':
            eval_suites[i] = EvaluationTournament(agent_path=apath, device='cpu')
            
    bots = {}
    for i, atype in enumerate(agent_types):
        if atype == 'heuristic':
            bots[i] = HeuristicBaseline(i)
            
    print(f"🎬 Starting {num_players}-player match: {agent_types}")
    
    done = False
    step = 0
    while not done and step < 500:
        actions = [[] for _ in range(num_players)]
        for pid in range(num_players):
            if obs_list[pid]['status'] != 'ACTIVE':
                continue
                
            raw_obs = obs_list[pid]['observation']
            if agent_types[pid] == 'model':
                p_obs = eval_suites[pid].obs_proc.process(raw_obs, player_id=pid)
                actions[pid] = eval_suites[pid].get_agent_actions(p_obs, raw_obs, player_id=pid)
            else:
                actions[pid] = bots[pid].act(raw_obs)
                
        obs_list = env.step(actions)
        done = any(state.get('status') != 'ACTIVE' for state in obs_list)
        step += 1
        if step % 50 == 0:
            print(f"Step {step}...")

    with open(output_path, "w") as f:
        f.write(env.render(mode="html", width=800, height=600))
    
    print(f"✅ Replay saved to: {output_path}")
    print(f"🏁 Final Outcome: {[s.get('reward', 0) for s in obs_list]}")

if __name__ == "__main__":
    # Match 1: 2 Heuristics
    print("\n--- MATCH 1: HEURISTIC vs HEURISTIC ---")
    render_match(['heuristic', 'heuristic'], [None, None], "heuristic_vs_heuristic.html")
    
    # Match 2: 2 BC Models
    print("\n--- MATCH 2: BC_PRETRAINED vs BC_PRETRAINED ---")
    bc_path = "checkpoints/bc_pretrained.pt"
    render_match(['model', 'model'], [bc_path, bc_path], "bc_vs_bc.html")
