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
        
        # 1 planet captured * 50.0 = 50.0. 
        # Time penalty -0.50.
        # Total = 49.5. Scale / 1000 = 0.0495.
        self.assertAlmostEqual(r, 0.0495, places=5)
        
        # Call 3: same state
        r2 = self.shaper.calculate_reward(obs2, False)
        # Deltas are 0. Time penalty -0.50 / 1000 = -0.0005
        self.assertAlmostEqual(r2, -0.0005, places=5)

    def test_production_swing(self):
        # Call 1: Initialization
        obs1 = {"planets": [[0, 0, 10, 10, 1, 100, 5]], "fleets": []} # My planet
        self.shaper.calculate_reward(obs1, False)

        # Call 2: Enemy gains a planet with 10 production
        obs2 = {
            "planets": [
                [0, 0, 10, 10, 1, 100, 5],
                [1, 1, 40, 40, 1, 100, 10]
            ],
            "fleets": []
        }
        r = self.shaper.calculate_reward(obs2, False)
        # Prod penalty: (10 - 0) * -100 = -1000.
        # Time penalty: -0.5.
        # Total = -1000.5. Scale / 1000 = -1.0005.
        self.assertAlmostEqual(r, -1.0005, places=5)

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
        # Terminal knockout reward: 1000 + (500 - 100) * 2 = 1800. Scale / 1000 = 1.8
        self.assertGreater(r, 1.5)

if __name__ == "__main__":
    unittest.main()
