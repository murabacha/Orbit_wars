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
        # NEW RULE: Hard Threshold is 15. No exceptions.
        # Source has 14 ships, trying to send 100% (14 ships) - All-in (4)
        obs = {
            "planets": [
                [0, 0, 10, 10, 1, 14, 5],
                [1, -1, 40, 40, 1, 10, 1]
            ],
            "fleets": []
        }
        # Target index 1, Allocation index 4 (100% = 14 ships)
        # Should be skipped because 14 < 15. No more exceptions for 4 and 5.
        moves = self.processor.process_actions(obs, 0, [1], [4])
        self.assertEqual(len(moves), 0)

        # Source has 60 ships, trying to send 25% (15 ships) - Bin (1)
        # Should work because 15 >= 15
        obs["planets"][0][5] = 60
        moves = self.processor.process_actions(obs, 0, [1], [1])
        self.assertEqual(len(moves), 1)

        # Source has 100 ships, trying to send 10% (wait, 25% is alloc 1)
        # Allocation 5 (Exact) with source 20, target 0. Should send ~5-10 ships.
        # Should be skipped because num_ships < 15.
        obs["planets"][0][5] = 20
        obs["planets"][1][5] = 0
        moves = self.processor.process_actions(obs, 0, [1], [5])
        self.assertEqual(len(moves), 0)

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

        # But allocation 5 (intercept) should still work for own planets (if ships >= 15)
        moves = self.processor.process_actions(obs, 0, [1], [5])
        self.assertEqual(len(moves), 1)

if __name__ == "__main__":
    unittest.main()
