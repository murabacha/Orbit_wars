import os
import sys
import torch
import numpy as np

# FIX: Robust path detection for Kaggle Environments where __file__ may not be defined.
try:
    PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments that use exec() (like some Kaggle versions)
    PACKAGE_PATH = "/kaggle_simulations/agent/"

if PACKAGE_PATH not in sys.path:
    sys.path.append(PACKAGE_PATH)

# Add current working directory as well, just in case
if "." not in sys.path:
    sys.path.append(".")

from model_def import TransformerPPOModel
from logic import ObservationProcessor, ActionProcessor, OrbitWarsWrapper

# Global state to persist model across steps
AGENT_STATE = {
    "model": None,
    "obs_proc": None,
    "act_proc": None,
    "wrapper": None,
    "player_id": None
}

def load_agent(obs, config):
    """
    Initializes the model and processors on the first step.
    """
    global AGENT_STATE
    
    device = torch.device("cpu")
    max_entities = 200
    feature_dim = 18
    
    # 1. Initialize Model
    model = TransformerPPOModel(
        feature_dim=feature_dim,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        max_entities=max_entities
    )
    
    # 2. Load Weights
    # Weights should be in the same folder as main.py
    model_path = os.path.join(PACKAGE_PATH, "model.pt")
    if not os.path.exists(model_path):
        # Local fallback for testing
        model_path = "model.pt"
        
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    # 3. Initialize Processors
    wrapper_config = {
        "shipSpeed": config.get("shipSpeed", 6.0),
        "sunRadius": config.get("sunRadius", 10.0),
        "boardSize": config.get("boardSize", 100.0),
        "episodeSteps": config.get("episodeSteps", 500)
    }
    wrapper = OrbitWarsWrapper(wrapper_config)
    obs_proc = ObservationProcessor(
        max_entities=max_entities,
        board_size=wrapper_config["boardSize"],
        max_speed=wrapper_config["shipSpeed"]
    )
    act_proc = ActionProcessor(wrapper)
    
    AGENT_STATE["model"] = model
    AGENT_STATE["obs_proc"] = obs_proc
    AGENT_STATE["act_proc"] = act_proc
    AGENT_STATE["wrapper"] = wrapper
    AGENT_STATE["player_id"] = obs.player

def agent(obs, config):
    """
    Kaggle Submission Entry Point.
    Processes the raw observation, runs inference, and returns processed actions.
    """
    global AGENT_STATE
    
    # Initialization
    if AGENT_STATE["model"] is None:
        load_agent(obs, config)
    
    player_id = AGENT_STATE["player_id"]
    model = AGENT_STATE["model"]
    obs_proc = AGENT_STATE["obs_proc"]
    act_proc = AGENT_STATE["act_proc"]
    wrapper = AGENT_STATE["wrapper"]
    
    # 1. Process Observation
    # The environment passed to the agent is the RAW dictionary-like object
    # We need to ensure it's compatible with our processor
    processed = obs_proc.process(obs, player_id=player_id)
    
    # 2. Generate Action Mask
    full_mask = wrapper.get_action_mask(obs, player_id=player_id)
    # Pad or truncate to max_entities
    max_entities = obs_proc.max_entities
    action_masks_grid = np.zeros((max_entities, max_entities), dtype=bool)
    n = min(full_mask.shape[0], max_entities)
    action_masks_grid[:n, :n] = full_mask[:n, :n]
    
    # 3. Inference
    with torch.no_grad():
        ent_t = torch.tensor(processed['entities'], dtype=torch.float32).unsqueeze(0)
        ids_t = torch.tensor(processed['entity_ids'], dtype=torch.long).unsqueeze(0)
        msk_t = torch.tensor(processed['mask'], dtype=torch.float32).unsqueeze(0)
        amsk_t = torch.tensor(action_masks_grid, dtype=torch.bool).unsqueeze(0)
        
        target_logits, alloc_logits, _ = model(ent_t, ids_t, msk_t, amsk_t)
        
        # Argmax for deterministic competitive play
        sampled_targets = target_logits.squeeze(0).argmax(dim=-1)
        
        B, N = 1, max_entities
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, N).reshape(-1)
        source_idx = torch.arange(N).unsqueeze(0).expand(B, -1).reshape(-1)
        
        selected_alloc_logits = alloc_logits.squeeze(0)[torch.arange(N), sampled_targets, :]
        sampled_allocs = selected_alloc_logits.argmax(dim=-1)
        
    # 4. Post-Process to Environment Format
    return act_proc.process_actions(
        obs, 
        player_id=player_id,
        target_indices=sampled_targets.cpu().numpy().tolist(),
        allocation_indices=sampled_allocs.cpu().numpy().tolist()
    )
