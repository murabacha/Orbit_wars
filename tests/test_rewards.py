import unittest
from environment.rewards import RewardShaper

class TestRewardShaper(unittest.TestCase):
    def setUp(self):
        self.shaper = RewardShaper(player_id=0)

    def test_reward_capture(self):
        # Call 1: Initialization
        obs1 = {"planets": [], "fleets": []}
        self.shaper.calculate_reward(obs1, False)
        
        # Call 2: Capture a 0-production asteroid
        obs2 = {"planets": [[0, 0, 50, 50, 1, 10, 0]], "fleets": []}
        r = self.shaper.calculate_reward(obs2, False)
        
        # 1 planet captured * 20.0 = 20.0. 
        # Ship advantage delta (10-0) - (0-0) = 10. 10 * 0.1 = 1.0.
        # Total = 21.0. Scale / 1000 = 0.021.
        # Time penalty -0.05 / 1000 = -0.00005.
        # Total approx 0.02095
        self.assertGreater(r, 0.02)
        
        # Call 3: same state
        r2 = self.shaper.calculate_reward(obs2, False)
        # Deltas are 0. Time penalty -0.05 / 1000 = -0.00005
        self.assertAlmostEqual(r2, -0.00005, places=5)

    def test_knockout_reward(self):
        # Call 1: Initialization
        obs1 = {"planets": [[0, 0, 10, 10, 1, 100, 5], [1, 1, 40, 40, 1, 10, 1]], "fleets": []}
        self.shaper.calculate_reward(obs1, False)

        # Player 0 has 1 planet, Player 1 has 0 planets (Knockout)
        obs2 = {
            "planets": [[0, 0, 10, 10, 1, 100, 5]], 
            "fleets": []
        }
        r = self.shaper.calculate_reward(obs2, True, episode_step=100)
        # Terminal knockout reward: 1000 + (500 - 100) = 1400. Scale / 1000 = 1.4
        self.assertGreater(r, 1.0)

    def test_timeout_win_reward(self):
        # Call 1: Initialization - both already have their planets
        obs1 = {
            "planets": [
                [0, 0, 10, 10, 1, 100, 5],
                [1, 1, 40, 40, 1, 100, 5]
            ],
            "fleets": []
        }
        self.shaper.calculate_reward(obs1, False)

        # Both have planets, but Player 0 has more ships
        obs2 = {
            "planets": [
                [0, 0, 10, 10, 1, 200, 5],
                [1, 1, 40, 40, 1, 100, 5]
            ],
            "fleets": []
        }
        r = self.shaper.calculate_reward(obs2, True)
        # Timeout win reward: 100. Scale / 1000 = 0.1
        # Advantage reward: (100 - 0) * 0.1 = 10. Scale / 1000 = 0.01
        # Total approx 0.11
        self.assertGreater(r, 0.1)

if __name__ == "__main__":
    unittest.main()
