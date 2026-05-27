
import sys
import os
from kaggle_environments import make

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.baseline.heuristic import HeuristicBaseline

def test_heuristic():
    env = make("orbit_wars", debug=True)
    
    # We'll use one HeuristicBaseline and one "random" agent
    hb = HeuristicBaseline(0)
    
    obs_list = env.reset()
    
    print("Starting verification of HeuristicBaseline (Elite)...")
    
    for step in range(10):
        # Player 0 uses our elite heuristic
        p0_obs = obs_list[0]['observation']
        moves_p0 = hb.act(p0_obs)
        
        # Player 1 uses random
        moves_p1 = "random"
        
        # Step the environment
        obs_list = env.step([moves_p0, moves_p1])
        
        print(f"Step {step}: Player 0 made {len(moves_p0)} moves.")
        
        done = any(state.get('status') != 'ACTIVE' for state in obs_list)
        if done:
            print("Game finished early.")
            break

    print("Verification complete. Elite Heuristic is OPERATIONAL.")

if __name__ == "__main__":
    test_heuristic()
