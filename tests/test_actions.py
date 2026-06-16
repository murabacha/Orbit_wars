import unittest
import math
from environment.wrapper import OrbitWarsWrapper
from environment.action_processor import ActionProcessor

class TestActionProcessor(unittest.TestCase):
    def setUp(self):
        self.wrapper = OrbitWarsWrapper({"shipSpeed": 6.0})
        self.processor = ActionProcessor(self.wrapper)

    def test_process_no_action(self):
        obs = {"planets": [[0, 0, 10, 10, 1, 10, 1]], "fleets": []}
        moves = self.processor.process_actions(obs, 0, [0], [0]) # allocation 0 = no action
        self.assertEqual(len(moves), 0)

    def test_batching_fix(self):
        # Source has 10 ships, trying to send 100% (10 ships)
        obs = {
            "planets": [
                [0, 0, 10, 10, 1, 10, 5],
                [1, -1, 40, 40, 1, 10, 1]
            ],
            "fleets": []
        }
        # Target index 1, Allocation index 4 (100% = 10 ships)
        moves = self.processor.process_actions(obs, 0, [1], [4])
        # Should be skipped because 10 < 15
        self.assertEqual(len(moves), 0)

        # Source has 100 ships, trying to send 25% (25 ships)
        obs["planets"][0][5] = 100
        moves = self.processor.process_actions(obs, 0, [1], [1])
        # Should NOT be skipped because 25 >= 15
        self.assertEqual(len(moves), 1)

    def test_shuffling_fix(self):
        # Target is OWN planet, and not using allocation 5 (intercept)
        obs = {
            "planets": [
                [0, 0, 10, 10, 1, 100, 5],
                [1, 0, 40, 40, 1, 10, 1] # Owned by player 0
            ],
            "fleets": []
        }
        # Allocation index 4 (100% = 100 ships)
        moves = self.processor.process_actions(obs, 0, [1], [4])
        # Should be skipped by shuffling fix
        self.assertEqual(len(moves), 0)

        # But allocation 5 (intercept) should still work for own planets (per existing logic)
        moves = self.processor.process_actions(obs, 0, [1], [5])
        self.assertEqual(len(moves), 1)

if __name__ == "__main__":
    unittest.main()
