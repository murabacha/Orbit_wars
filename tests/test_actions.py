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
        # NEW RULE: Threshold is 10.
        # Source has 9 ships, trying to send 100% (9 ships) - All-in (4)
        obs = {
            "planets": [
                [0, 0, 10, 10, 1, 9, 5],
                [1, -1, 40, 40, 1, 10, 1]
            ],
            "fleets": []
        }
        # Target index 1, Allocation index 4 (100% = 9 ships)
        # Exception: 4 and 5 are allowed even if < 10.
        moves = self.processor.process_actions(obs, 0, [1], [4])
        self.assertEqual(len(moves), 1)

        # Source has 36 ships, trying to send 25% (9 ships) - Bin (1)
        # Should be skipped because 9 < 10 and alloc_idx (1) is not in [4, 5]
        obs["planets"][0][5] = 36
        moves = self.processor.process_actions(obs, 0, [1], [1])
        self.assertEqual(len(moves), 0)

        # Source has 100 ships, trying to send 25% (25 ships)
        obs["planets"][0][5] = 100
        moves = self.processor.process_actions(obs, 0, [1], [1])
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

        # But allocation 5 (intercept) should still work for own planets
        moves = self.processor.process_actions(obs, 0, [1], [5])
        self.assertEqual(len(moves), 1)

if __name__ == "__main__":
    unittest.main()
