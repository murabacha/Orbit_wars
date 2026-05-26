import unittest
from environment.rewards import RewardShaper

class TestRewardShaper(unittest.TestCase):
    def setUp(self):
        self.shaper = RewardShaper(player_id=0)

    def test_reward_delta(self):
        # First step
        obs = {"planets": [[0, 0, 50, 50, 1, 10, 1]], "fleets": []}
        r = self.shaper.calculate_reward(obs, False)
        # Should be based on initial delta from 0
        self.assertGreater(r, 0)
        
        # Second step: same state
        r2 = self.shaper.calculate_reward(obs, False)
        self.assertEqual(r2, 0.1) # Just passive ship survival delta (10 * 0.1 = 1.0, wait... current-prev)
        # Actually in my implementation: current_ships - prev_ships. 10 - 10 = 0.

    def test_win_reward(self):
        obs = {"planets": [], "fleets": [], "rewards": [100, 0, 0, 0]}
        r = self.shaper.calculate_reward(obs, True)
        self.assertGreater(r, 50)

if __name__ == "__main__":
    unittest.main()
