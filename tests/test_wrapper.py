import unittest
import math
from environment.wrapper import OrbitWarsWrapper
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

class TestOrbitWarsWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = OrbitWarsWrapper({"shipSpeed": 6.0, "sunRadius": 10.0})

    def test_calculate_speed(self):
        self.assertEqual(self.wrapper.calculate_speed(1), 1.0)
        self.assertGreater(self.wrapper.calculate_speed(1000), 5.0)
        self.assertEqual(self.wrapper.calculate_speed(2000), 6.0)

    def test_intercept_static(self):
        # Source (0,0), Target (50,0) static
        tgt_data = {
            'x': 50, 'y': 0, 'id': 1, 'owner': -1, 'production': 1, 'ships': 10, 'source_ships': 100
        }
        angle, time, tx, ty = self.wrapper.get_intercept_params((0, 0), tgt_data, 1.0, {})
        
        self.assertAlmostEqual(angle, 0.0)
        self.assertGreater(time, 0)
        self.assertEqual(tx, 50)

    def test_sun_safety(self):
        # Path from (20,20) to (80,80) goes through sun at (50,50)
        safe = self.wrapper.is_path_safe(20, 20, math.pi/4, 100)
        self.assertFalse(safe)
        
        # Path from (0,0) to (10,10) is safe
        safe = self.wrapper.is_path_safe(0, 0, math.pi/4, 5)
        self.assertTrue(safe)

if __name__ == "__main__":
    unittest.main()
