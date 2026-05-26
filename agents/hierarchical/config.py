from dataclasses import dataclass

@dataclass
class HierarchicalConfig:
    # Macro Settings
    input_dim: int = 24 # Expanded global features
    num_modes: int = 4
    macro_interval: int = 10 # Decision every 10 turns
    
    # Training
    learning_rate: float = 1e-4
    gamma: float = 0.999 # Very long horizon
    
    # Environment
    player_id: int = 0
