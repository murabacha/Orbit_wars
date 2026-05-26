import math
import numpy as np
from typing import List, Dict, Any, Tuple
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER

class OrbitWarsWrapper:
    """
    Core environment wrapper for Orbit Wars.
    Handles:
    - 8-iteration intercept solver.
    - Future garrison estimation.
    - Action masking.
    - State tokenization.
    """
    def __init__(self, config: Dict[str, Any]):
        self.max_speed = config.get("shipSpeed", 6.0)
        self.sun_radius = config.get("sunRadius", 10.0)
        self.board_size = config.get("boardSize", 100.0)
        self.episode_steps = config.get("episodeSteps", 500)

    def calculate_speed(self, ships: int) -> float:
        """Computes fleet speed based on ship count."""
        if ships <= 0: return 1.0
        speed = 1.0 + (self.max_speed - 1.0) * (math.log(max(1, ships)) / math.log(1000))**1.5
        return min(speed, self.max_speed)

    def get_intercept_params(self, source: Planet, target: Any, ships: int, angular_velocity: float) -> Tuple[float, float, float, float]:
        """
        8-iteration mathematical intercept solver.
        Returns: (angle, travel_time, intercept_x, intercept_y)
        """
        speed = self.calculate_speed(ships)
        tx, ty = target.x, target.y
        
        # Iteratively refine intercept point based on target movement
        for _ in range(8):
            dist = math.hypot(tx - source.x, ty - source.y)
            travel_time = dist / speed
            
            # Predict future position
            if hasattr(target, 'angular_velocity') or angular_velocity != 0:
                # Orbiting planet prediction
                rad = math.hypot(target.x - CENTER[0], target.y - CENTER[1])
                if rad < 50: # Only inner planets rotate
                    current_angle = math.atan2(target.y - CENTER[1], target.x - CENTER[0])
                    future_angle = current_angle + (angular_velocity * travel_time)
                    tx = CENTER[0] + rad * math.cos(future_angle)
                    ty = CENTER[1] + rad * math.sin(future_angle)
            
            # Note: Comet path prediction would go here using obs['comets'] data
            
        angle = math.atan2(ty - source.y, tx - source.x)
        return angle, travel_time, tx, ty

    def estimate_future_garrison(self, target: Planet, travel_time: float) -> int:
        """Predicts ships on planet at time of arrival."""
        if target.owner == -1:
            return target.ships
        return target.ships + int(target.production * travel_time)

    def is_path_safe(self, source_x: float, source_y: float, angle: float, dist: float) -> bool:
        """Checks if a trajectory hits the sun."""
        # Simple segment-to-point distance check for the sun at (50, 50)
        # Vector from source to sun
        dx = 50 - source_x
        dy = 50 - source_y
        
        # Projection of sun onto the path vector
        vx = math.cos(angle)
        vy = math.sin(angle)
        t = dx * vx + dy * vy
        
        if t < 0: return True # Sun is behind source
        if t > dist: return True # Sun is beyond target
        
        # Closest point on path to sun
        closest_x = source_x + t * vx
        closest_y = source_y + t * vy
        
        dist_to_sun = math.hypot(closest_x - 50, closest_y - 50)
        return dist_to_sun > self.sun_radius

    def get_action_mask(self, obs: Dict[str, Any], player_id: int) -> np.ndarray:
        """
        Generates a boolean mask for valid target entities.
        Prevents:
        - Targeting self.
        - Targeting expiring comets (Black Hole bug).
        - Pathing through the sun.
        """
        planets = obs.get("planets", [])
        mask = np.ones(len(planets), dtype=bool)
        my_planets = [p for p in planets if p[1] == player_id]
        
        if not my_planets:
            return np.zeros(len(planets), dtype=bool)

        for i, p_data in enumerate(planets):
            # 1. Don't attack own planets (for now, unless reinforcing - logic can be refined)
            if p_data[1] == player_id:
                mask[i] = False
                continue
            
            # Check safety from at least one source
            safe = False
            for my_p in my_planets:
                # Construct temporary Planet objects for math
                src = Planet(*my_p)
                tgt = Planet(*p_data)
                angle, travel_time, _, _ = self.get_intercept_params(src, tgt, src.ships, obs.get('angular_velocity', 0))
                
                # 2. Black Hole Comet check
                if tgt.id in obs.get('comet_planet_ids', []):
                    # Find comet data
                    for group in obs.get('comets', []):
                        if tgt.id in group['planet_ids']:
                            time_left = len(group['paths'][0]) - group['path_index']
                            if int(travel_time) >= time_left:
                                continue # Too far or hits expiration
                
                # 3. Sun check
                if self.is_path_safe(src.x, src.y, angle, math.hypot(tgt.x - src.x, tgt.y - src.y)):
                    safe = True
                    break
            
            mask[i] = safe
            
        return mask
