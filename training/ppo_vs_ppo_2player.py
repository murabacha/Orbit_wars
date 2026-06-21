
import os
import sys
import torch
from kaggle_environments import make

# Enforce clean path insertions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.render_custom_match import render_match

if __name__ == "__main__":
    # Prioritize checking the submission folder for downloaded weights, fallback to latest checkpoint
    checkpoint1 = "../submission/model.pt" if os.path.exists("../submission/model.pt") else "submission/model.pt"
    if not os.path.exists(checkpoint1):
        checkpoint1 = "checkpoints/bc_pretrained2.pt"
        
    checkpoint2 = "checkpoints/bc_pretrained2.pt"
    
    # Check if files exist
    for cp in [checkpoint1, checkpoint2]:
        if not os.path.exists(cp):
            # Try absolute path or relative to project root if script is run from Orbit_wars/training
            if not os.path.exists(os.path.join("..", cp)):
                 print(f"Error: {cp} not found.")
                 sys.exit(1)

    print(f"⚔️ 2-Player Match between {checkpoint1} and {checkpoint2}")
    
    # Run the match with only 2 players
    # Using relative paths from Orbit_wars directory
    render_match(
        agent_types=['model', 'model'],
        agent_paths=[checkpoint1, checkpoint2],
        output_path="ppo_vs_ppo_2player.html"
    )
    
    print("\nColor assignment:")
    print(f"🔵 BLUE: {checkpoint1}")
    print(f"🔴 RED:  {checkpoint2}")
    print("\nResult saved to ppo_vs_ppo_2player.html")
