import math
import numpy as np
from typing import List, Dict, Any, Tuple
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER

class OrbitWarsWrapper:
    """
    Refactored, mathematically accurate environment wrapper for Orbit Wars.
    Implements:
    - 8-iteration intercept solver with loop-drift prevention.
    - Dynamic allocation-aware speed calculation.
    - Individual planet/comet path prediction.
    - Sun-collision path safety.
    """
    def __init__(self, config: Dict[str, Any]):
        self.max_speed = config.get("shipSpeed", 6.0)
        self.sun_radius = config.get("sunRadius", 10.0)
        self.board_size = config.get("boardSize", 100.0)
        self.episode_steps = config.get("episodeSteps", 500)

    def calculate_speed(self, ships: int) -> float:
        """
        Computes precise logarithmic fleet speed based on ship count.
        Speed = 1.0 + (max_speed - 1.0) * (log(max(1, ships)) / log(1000))^1.5
        """
        if ships <= 0: return 1.0
        log_ratio = math.log(max(1, ships)) / math.log(1000)
        speed = 1.0 + (self.max_speed - 1.0) * (log_ratio ** 1.5)
        return min(speed, self.max_speed)

    def get_intercept_params(self, source_pos: Tuple[float, float], target_data: Dict[str, Any], allocation_percentage: float, obs: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        8-iteration mathematical intercept solver.
        - source_pos: (x, y) tuple.
        - target_data: Dictionary containing x, y, id, owner, production, ships.
        """
        source_x, source_y = source_pos
        
        # 2. Calculate speed based on dynamic allocation
        # We need to know source_ships to calculate absolute ships sent
        source_ships = target_data.get('source_ships', 100) # Fallback or passed in dict
        ships_sent = int(source_ships * allocation_percentage)
        speed = self.calculate_speed(ships_sent)
        
        tx, ty = target_data['x'], target_data['y']
        travel_time = 0.0
        
        # 8-iteration refinement loop
        for _ in range(8):
            dist = math.hypot(tx - source_x, ty - source_y)
            travel_time = dist / speed
            
            # Predict future position using individual entity data
            tx, ty = self.predict_future_position(target_data, travel_time, obs)
            
        angle = math.atan2(ty - source_y, tx - source_x)
        return angle, travel_time, tx, ty

    def predict_future_position(self, target_data: Dict[str, Any], travel_time: float, obs: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predicts future coordinates of planets or comets using individual data structures.
        """
        target_id = target_data['id']
        
        # A. Comet Path Data Check
        for group in obs.get('comets', []):
            if target_id in group['planet_ids']:
                idx = group['planet_ids'].index(target_id)
                path = group['paths'][idx]
                curr_idx = group['path_index']
                # Index directly into pre-calculated trajectory
                future_idx = min(len(path) - 1, int(curr_idx + travel_time))
                return path[future_idx][0], path[future_idx][1]
        
        # B. Orbiting Planet Logic (Individual Data)
        dx = target_data['x'] - CENTER
        dy = target_data['y'] - CENTER
        rad = math.hypot(dx, dy)
        
        if rad < 45.0: # Inner orbiting planets
            # Use per-planet angular velocity from observation or object data
            angular_velocity = obs.get('planet_angular_velocities', {}).get(target_id, 0.0)
            if angular_velocity == 0:
                # Fallback to attribute check if not in obs
                angular_velocity = target_data.get('angular_velocity', 0.02)
            
            # Always start from initial constants to prevent cumulative drift
            initial_angle = math.atan2(dy, dx)
            future_angle = initial_angle + (angular_velocity * travel_time)
            
            tx = CENTER + rad * math.cos(future_angle)
            ty = CENTER + rad * math.sin(future_angle)
            return tx, ty
            
        # C. Stationary Planet
        return target_data['x'], target_data['y']

    def estimate_future_garrison(self, target_data: Dict[str, Any], travel_time: float) -> int:
        """Predicts ships on planet at time of arrival."""
        if target_data['owner'] == -1: # Neutral
            return target_data['ships']
        return target_data['ships'] + int(target_data['production'] * travel_time)

    def is_path_safe(self, source_x: float, source_y: float, angle: float, dist: float) -> bool:
        """Checks if a trajectory collides with the sun at (50, 50)."""
        dx = CENTER - source_x
        dy = CENTER - source_y
        
        vx, vy = math.cos(angle), math.sin(angle)
        t = dx * vx + dy * vy
        
        # Clamp to segment [0, dist]
        t = max(0.0, min(dist, t))
        
        closest_x = source_x + t * vx
        closest_y = source_y + t * vy
        
        dist_to_sun = math.hypot(closest_x - CENTER, closest_y - CENTER)
        return dist_to_sun > self.sun_radius

    def get_action_mask(self, obs: Dict[str, Any], player_id: int, allocation_percentage: float = 1.0) -> np.ndarray:
        """
        Generates a target entity mask using dynamic allocation_percentage for speed calculation.
        Prevents targeting expiring comets and sun collisions.
        """
        planets_raw = obs.get("planets", [])
        mask = np.ones(len(planets_raw), dtype=bool)
        
        planets = [Planet(*p) for p in planets_raw]
        my_planets = [p for p in planets if p.owner == player_id]
        
        if not my_planets:
            return np.zeros(len(planets), dtype=bool)

        comet_ids = obs.get('comet_planet_ids', [])

        for i, target in enumerate(planets):
            if target.owner == player_id:
                mask[i] = False
                continue
            
            target_data = {
                'x': target.x,
                'y': target.y,
                'id': target.id,
                'owner': target.owner,
                'production': target.production,
                'ships': target.ships
            }

            safe_path_exists = False
            for source in my_planets:
                if source.ships < 1: continue
                
                target_data['source_ships'] = source.ships
                # Calculate intercept with dynamic allocation speed
                angle, travel_time, tx, ty = self.get_intercept_params((source.x, source.y), target_data, allocation_percentage, obs)
                
                # Black Hole Comet check (Race condition on expiration turn)
                if target.id in comet_ids:
                    invalid_arrival = False
                    for group in obs.get('comets', []):
                        if target.id in group['planet_ids']:
                            turns_left = len(group['paths'][0]) - group['path_index']
                            if int(travel_time) >= turns_left:
                                invalid_arrival = True
                                break
                    if invalid_arrival: continue
                
                # Path safety check
                total_dist = math.hypot(tx - source.x, ty - source.y)
                if self.is_path_safe(source.x, source.y, angle, total_dist):
                    safe_path_exists = True
                    break
            
            mask[i] = safe_path_exists
            
        return mask
