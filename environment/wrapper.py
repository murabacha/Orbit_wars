import math
import numpy as np
from typing import List, Dict, Any, Tuple
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER

class OrbitWarsWrapper:
    """
    Refactored environment wrapper for Orbit Wars.
    Correctly maps Planet attributes [id, owner, x, y, radius, ships, production].
    """
    def __init__(self, config: Dict[str, Any]):
        self.max_speed = config.get("shipSpeed", 6.0)
        self.sun_radius = config.get("sunRadius", 10.0)
        self.board_size = config.get("boardSize", 100.0)
        self.episode_steps = config.get("episodeSteps", 500)

    def calculate_speed(self, ships: int) -> float:
        if ships <= 0: return 1.0
        log_ratio = math.log(max(1, ships)) / math.log(1000)
        speed = 1.0 + (self.max_speed - 1.0) * (log_ratio ** 1.5)
        return min(speed, self.max_speed)

    def get_intercept_params(self, source_pos: Tuple[float, float], target_data: Dict[str, Any], allocation_percentage: float, obs: Dict[str, Any]) -> Tuple[float, float, float, float]:
        source_x, source_y = source_pos
        source_ships = target_data.get('source_ships', 100)
        ships_sent = int(source_ships * allocation_percentage)
        speed = self.calculate_speed(ships_sent)
        tx, ty = target_data['x'], target_data['y']
        travel_time = 0.0
        for _ in range(8):
            dist = math.hypot(tx - source_x, ty - source_y)
            travel_time = dist / speed
            tx, ty = self.predict_future_position(target_data, travel_time, obs)
        angle = math.atan2(ty - source_y, tx - source_x)
        return angle, travel_time, tx, ty

    def predict_future_position(self, target_data: Dict[str, Any], travel_time: float, obs: Dict[str, Any]) -> Tuple[float, float]:
        target_id = target_data['id']
        for group in obs.get('comets', []):
            if target_id in group['planet_ids']:
                idx = group['planet_ids'].index(target_id)
                path = group['paths'][idx]
                curr_idx = group['path_index']
                future_idx = min(len(path) - 1, int(curr_idx + travel_time))
                return path[future_idx][0], path[future_idx][1]
        dx, dy = target_data['x'] - CENTER, target_data['y'] - CENTER
        rad = math.hypot(dx, dy)
        if rad < 45.0:
            angular_velocity = obs.get('planet_angular_velocities', {}).get(target_id, target_data.get('angular_velocity', 0.02))
            initial_angle = math.atan2(dy, dx)
            future_angle = initial_angle + (angular_velocity * travel_time)
            return CENTER + rad * math.cos(future_angle), CENTER + rad * math.sin(future_angle)
        return target_data['x'], target_data['y']

    def estimate_future_garrison(self, target_data: Dict[str, Any], travel_time: float) -> int:
        if target_data['owner'] == -1: return target_data['ships']
        return target_data['ships'] + int(target_data['production'] * travel_time)

    def is_path_safe(self, source_x: float, source_y: float, angle: float, dist: float) -> bool:
        dx, dy = CENTER - source_x, CENTER - source_y
        vx, vy = math.cos(angle), math.sin(angle)
        t = max(0.0, min(dist, dx * vx + dy * vy))
        closest_x, closest_y = source_x + t * vx, source_y + t * vy
        return math.hypot(closest_x - CENTER, closest_y - CENTER) > self.sun_radius

    def get_action_mask(self, obs: Dict[str, Any], player_id: int, allocation_percentage: float = 1.0) -> np.ndarray:
        planets_raw = obs.get("planets", [])
        mask = np.ones(len(planets_raw), dtype=bool)
        planets = [Planet(*p[:7]) for p in planets_raw]
        my_planets = [p for p in planets if p.owner == player_id]
        if not my_planets: return np.zeros(len(planets), dtype=bool)
        comet_ids = obs.get('comet_planet_ids', [])
        for i, target in enumerate(planets):
            if target.owner == player_id:
                mask[i] = False
                continue
            target_data = {'x': target.x, 'y': target.y, 'id': target.id, 'owner': target.owner, 'production': target.production, 'ships': target.ships}
            safe_path_exists = False
            for source in my_planets:
                if source.ships < 1: continue
                target_data['source_ships'] = source.ships
                angle, travel_time, tx, ty = self.get_intercept_params((source.x, source.y), target_data, allocation_percentage, obs)
                if target.id in comet_ids:
                    invalid_arrival = False
                    for group in obs.get('comets', []):
                        if target.id in group['planet_ids']:
                            if int(travel_time) >= len(group['paths'][0]) - group['path_index']:
                                invalid_arrival = True
                                break
                    if invalid_arrival: continue
                if self.is_path_safe(source.x, source.y, angle, math.hypot(tx - source.x, ty - source.y)):
                    safe_path_exists = True
                    break
            mask[i] = safe_path_exists
        return mask
