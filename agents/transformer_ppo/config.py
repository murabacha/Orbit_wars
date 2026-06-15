from dataclasses import dataclass

@dataclass
class TransformerPPOConfig:
    # Architecture
    feature_dim: int = 18
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    max_entities: int = 200
    
    # Training Hyperparameters
    learning_rate: float = 5e-6
    gamma: float = 0.998 # Long horizon focus
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # PPO Specifics
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    
    # Environment
    player_id: int = 0
    total_timesteps: int = 10_000_000
