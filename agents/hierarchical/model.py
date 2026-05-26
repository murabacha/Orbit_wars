import torch
import torch.nn as nn

class MacroManagerModel(nn.Module):
    """
    High-level Manager for Hierarchical RL.
    Selects a strategic mode based on aggregated global statistics.
    """
    def __init__(self, input_dim: int = 20, num_modes: int = 4):
        super().__init__()
        # Input features: [MyShips, EnemyShips, MyProd, EnemyProd, MyPlanets, EnemyPlanets, ...]
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Policy: Macro Mode Logits
        # Modes: 0=Expansion, 1=Consolidation, 2=Aggression, 3=CometFocus
        self.actor = nn.Linear(64, num_modes)
        
        # Value: State Value
        self.critic = nn.Linear(64, 1)

    def forward(self, global_stats: torch.Tensor):
        x = self.network(global_stats)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
