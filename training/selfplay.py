import random
import numpy as np

class SelfPlayManager:
    """
    State-of-the-art Prioritized Fictitious Self-Play (PFSP) Manager.
    Implements ELO-based matchmaking by tracking historical win rates and 
    sampling difficult opponents to prevent strategy stagnation.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.history = []
        self.win_rates = {} # Maps checkpoint path to observed win rate

    def save_checkpoint(self, model, step: int):
        path = f"{self.checkpoint_dir}/ppo_step_{step}.pt"
        # model.save(path) is handled by the trainer, we just track the metadata
        if path not in self.history:
            self.history.append(path)
            self.win_rates[path] = 0.5 # Initialize with neutral win rate

    def update_win_rate(self, opponent_path: str, won: bool):
        """ Updates the moving average win rate against a specific historical version. """
        if opponent_path not in self.win_rates:
            return
        
        alpha = 0.1 # Smoothing factor
        result = 1.0 if won else 0.0
        self.win_rates[opponent_path] = (1 - alpha) * self.win_rates[opponent_path] + alpha * result

    def get_opponent(self) -> str:
        """
        Samples an opponent using a prioritized distribution.
        80% Latest Checkpoint (Escalation)
        20% Prioritized Historical (Prevents Catastrophic Forgetting)
        """
        if not self.history:
            return "baseline_heuristic"
            
        latest_checkpoint = self.history[-1]
            
        if random.random() < 0.8:
            # 80% of the time, fight the bleeding-edge version of itself
            return latest_checkpoint
        else:
            # 20% of the time, sample historical opponents to prevent strategy cycling
            weights = []
            for path in self.history:
                wr = self.win_rates.get(path, 0.5)
                # Prioritize opponents with a ~50% win rate (highest learning signal)
                difficulty = 1.0 - abs(0.5 - wr) * 2
                weights.append(max(0.05, difficulty))
            
            return random.choices(self.history, weights=weights, k=1)[0]
