"""
Standalone Battle Royale Script
Runs a full 4-player match with the current bc_pretrained checkpoint.
"""
import os
import sys
import torch
import numpy as np
from kaggle_environments import make

# Add CWD to path to find local modules
sys.path.insert(0, os.path.abspath('.'))

from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel
from orbit_wars_ai.environment.observation_processor import ObservationProcessor
from orbit_wars_ai.environment.action_processor import ActionProcessor
from orbit_wars_ai.environment.wrapper import OrbitWarsWrapper

def run_battle_royale(checkpoint_path='checkpoints/bc_pretrained1.pt', output_path='bc_battle_royale.html'):
    device = 'cpu'
    max_entities = 200
    
    # 1. Load Model
    model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=max_entities)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded weights from: {checkpoint_path}")
    else:
        print(f"❌ ERROR: {checkpoint_path} not found!")
        return

    model.eval()

    # 2. Setup Components
    wrapper_config = {'shipSpeed': 6.0, 'sunRadius': 10.0, 'boardSize': 100.0, 'episodeSteps': 500}
    wrapper = OrbitWarsWrapper(wrapper_config)
    obs_proc = ObservationProcessor(max_entities=max_entities, board_size=100.0, max_speed=6.0)
    act_proc = ActionProcessor(wrapper)

    # 3. Define Agent Logic
    def bc_agent(obs):
        player_id = obs['player']
        p_obs = obs_proc.process(obs, player_id=player_id)
        
        full_mask = wrapper.get_action_mask(obs, player_id=player_id)
        action_masks_grid = np.zeros((max_entities, max_entities), dtype=bool)
        action_masks_grid[:full_mask.shape[0], :full_mask.shape[1]] = full_mask

        with torch.no_grad():
            ent_t = torch.tensor(p_obs['entities'], dtype=torch.float32).unsqueeze(0).to(device)
            ids_t = torch.tensor(p_obs['entity_ids'], dtype=torch.long).unsqueeze(0).to(device)
            msk_t = torch.tensor(p_obs['mask'], dtype=torch.float32).unsqueeze(0).to(device)
            amsk_t = torch.tensor(action_masks_grid, dtype=torch.bool).unsqueeze(0).to(device)

            target_logits, alloc_logits, _ = model(ent_t, ids_t, msk_t, amsk_t)
            
            # Use argmax for best deterministic behavior
            sampled_targets = target_logits.squeeze(0).argmax(dim=-1)
            batch_idx = torch.arange(max_entities, device=device)
            selected_alloc_logits = alloc_logits.squeeze(0)[batch_idx, sampled_targets, :]
            sampled_allocs = selected_alloc_logits.argmax(dim=-1)

        return act_proc.process_actions(
            obs, player_id=player_id,
            target_indices=sampled_targets.cpu().numpy().tolist(),
            allocation_indices=sampled_allocs.cpu().numpy().tolist()
        )

    # 4. Run Match
    env = make('orbit_wars', debug=True)
    print("🚀 Starting full 4-player Battle Royale (500 steps). Please wait...")
    
    # Run the simulation
    env.run([bc_agent, bc_agent, bc_agent, bc_agent])
    
    # 5. Save and Summarize
    with open(output_path, 'w') as f:
        f.write(env.render(mode='html'))
    
    print(f"🏁 Match Finished.")
    print(f"✅ Replay saved to: {output_path}")
    
    final_rewards = [s.get('reward', 0) for s in env.steps[-1]]
    print(f"Final Scores: {final_rewards}")

if __name__ == "__main__":
    run_battle_royale()
