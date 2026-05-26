class SelfPlayManager:
    """
    Manages the pool of opponent agents for League Training.
    Saves checkpoints and selects previous versions for the agent to play against.
    """
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.history = []

    def save_checkpoint(self, model, step: int):
        path = f"{self.checkpoint_dir}/agent_{step}.pt"
        # torch.save(model.state_dict(), path)
        self.history.append(path)

    def get_opponent(self, strategy: str = "pfsp"):
        """
        Prioritized Fictitious Self-Play (PFSP) selection.
        """
        if not self.history:
            return "baseline_heuristic"
        # Select a random historical checkpoint to prevent strategy forgetting
        import random
        return random.choice(self.history)
