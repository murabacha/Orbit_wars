import sys
from orbit_wars_ai.agents.baseline.heuristic import HeuristicBaseline

# This is the entry point for the Kaggle submission.
# It should bundle the model and policy into a single executable function.

def agent(obs, config):
    """
    Kaggle-compatible agent interface.
    """
    player_id = obs.player
    
    # For now, we use the heuristic as a placeholder in main.py
    # In a full deployment, this would load the Transformer weights and run inference.
    baseline = HeuristicBaseline(player_id)
    return baseline.act(obs)
