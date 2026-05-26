import torch
import torch.distributions as dist
from .model import MacroManagerModel
from .config import HierarchicalConfig

class HierarchicalPolicy:
    """
    Policy that manages macro-mode selection and hands off to low-level workers.
    """
    def __init__(self, config: HierarchicalConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.manager = MacroManagerModel(
            input_dim=config.input_dim,
            num_modes=config.num_modes
        ).to(device)
        self.current_mode = 0
        self.turn_counter = 0

    def get_macro_action(self, global_stats: torch.Tensor):
        if self.turn_counter % self.config.macro_interval == 0:
            with torch.no_grad():
                logits, value = self.manager(global_stats.to(self.device))
                categorical = dist.Categorical(logits=logits)
                self.current_mode = categorical.sample().item()
        
        self.turn_counter += 1
        return self.current_mode

    def execute_mode(self, mode: int, obs: dict):
        """
        Translates macro mode into low-level planet/fleet commands.
        (Typically calls a rule-based worker or a specialized sub-policy).
        """
        # Example pseudo-logic:
        # if mode == 0: return self.worker_expand(obs)
        # elif mode == 1: return self.worker_defend(obs)
        return [] # Default no-op
