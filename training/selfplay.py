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
        Higher priority is given to opponents where win_rate is closest to 50%.
        """
        if not self.history:
            return "baseline_heuristic"
            
        # Select components of the PFSP distribution
        # 80% chance: Prioritized sampling based on difficulty
        # 20% chance: Uniform random sampling to maintain general robustness
        if random.random() < 0.8:
            # Calculate weights based on proximity to 0.5 win rate (the "sweet spot" for learning)
            # Opponents we beat 100% or 0% of the time provide less gradient signal
            weights = []
            for path in self.history:
                wr = self.win_rates.get(path, 0.5)
                # Difficulty weight: 1.0 - abs(0.5 - wr) * 2
                # This peaks at 1.0 when wr is 0.5 and drops to 0.0 at 0.0 or 1.0
                difficulty = 1.0 - abs(0.5 - wr) * 2
                weights.append(max(0.05, difficulty))
            
            return random.choices(self.history, weights=weights, k=1)[0]
        else:
            return random.choice(self.history)
