import unittest
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

    def test_process_valid_move(self):
        # My planet ID 0, Target planet ID 1
        obs = {
            "planets": [
                [0, 0, 10, 10, 1, 100, 5],
                [1, -1, 40, 40, 1, 10, 1]
            ],
            "fleets": []
        }
        # Target index 1, Allocation index 4 (100%)
        moves = self.processor.process_actions(obs, 0, [1], [4])
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][0], 0) # Source ID
        self.assertEqual(moves[0][2], 100) # Ships

if __name__ == "__main__":
    unittest.main()
