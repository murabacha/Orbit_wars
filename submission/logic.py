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
        self.max_entities = config.get("max_entities", 200)

    def calculate_speed(self, ships: int) -> float:
        if ships <= 0: return 1.0
        log_ratio = math.log(max(1, ships)) / math.log(1000)
        speed = 1.0 + (self.max_speed - 1.0) * (log_ratio ** 1.5)
        return min(speed, self.max_speed)

    def get_intercept_params(self, source_pos: Tuple[float, float], source_radius: float, target_data: Dict[str, Any], allocation_percentage: float, obs: Dict[str, Any]) -> Tuple[float, float, float, float]:
        source_x, source_y = source_pos
        source_ships = target_data.get('source_ships', 100)
        target_radius = target_data.get('radius', 0.0)
        ships_sent = int(source_ships * allocation_percentage)
        speed = self.calculate_speed(ships_sent)
        tx, ty = target_data['x'], target_data['y']
        travel_time = 0.0
        for _ in range(8):
            # FIX: Surface-to-Surface distance (matches elite_heuristic launch physics)
            raw_dist = math.hypot(tx - source_x, ty - source_y)
            dist = max(0.0, raw_dist - source_radius - target_radius - 0.1)
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
        
        # Use board_size instead of imported CENTER to prevent tuple TypeError
        center = self.board_size / 2.0 
        dx, dy = target_data['x'] - center, target_data['y'] - center
        rad = math.hypot(dx, dy)
        target_radius = target_data.get('radius', 0.0)
        
        # FIX: Align orbit check with elite_heuristic's ROTATION_LIMIT
        if rad + target_radius < 50.0:
            angular_velocity = obs.get('planet_angular_velocities', {}).get(target_id, target_data.get('angular_velocity', 0.02))
            initial_angle = math.atan2(dy, dx)
            future_angle = initial_angle + (angular_velocity * travel_time)
            return center + rad * math.cos(future_angle), center + rad * math.sin(future_angle)
        return target_data['x'], target_data['y']

    def estimate_future_garrison(self, target_data: Dict[str, Any], travel_time: float) -> int:
        if target_data['owner'] == -1: return target_data['ships']
        return target_data['ships'] + int(target_data['production'] * travel_time)

    def is_path_safe(self, source_x: float, source_y: float, angle: float, dist: float) -> bool:
        # FIX: Define center as a float to prevent tuple TypeErrors
        center_val = self.board_size / 2.0 
        
        dx, dy = center_val - source_x, center_val - source_y
        vx, vy = math.cos(angle), math.sin(angle)
        t = max(0.0, min(dist, dx * vx + dy * vy))
        closest_x, closest_y = source_x + t * vx, source_y + t * vy
        
        return math.hypot(closest_x - center_val, closest_y - center_val) > self.sun_radius

    def get_action_mask(self, obs: Dict[str, Any], player_id: int, allocation_percentage: float = 1.0) -> np.ndarray:
        planets_raw = obs.get("planets", [])
        fleets_raw = obs.get("fleets", [])
        actual_entity_count = len(planets_raw) + len(fleets_raw)
        
        # 1. Mask out invalid padding sources and targets
        mask = np.zeros((self.max_entities, self.max_entities), dtype=bool)
        
        planets = [Planet(*p[:7]) for p in planets_raw]
        comet_ids = obs.get('comet_planet_ids', [])
        
        for s_idx, source in enumerate(planets):
            if s_idx >= self.max_entities: break
            if source.owner != player_id or source.ships < 1:
                continue
            
            # Always allow self-targeting as a "do-nothing" action to prevent uniform trap
            mask[s_idx, s_idx] = True
            
            # REMOVED MIN_STRIKE_FORCE: Restoring full strategic autonomy to the agent.
                
            for t_idx, target in enumerate(planets):
                if t_idx >= self.max_entities: break
                if t_idx == s_idx:
                    continue
                
                target_data = {'x': target.x, 'y': target.y, 'radius': target.radius, 'id': target.id, 'owner': target.owner, 'production': target.production, 'ships': target.ships, 'source_ships': source.ships}
                
                angle, travel_time, tx, ty = self.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_percentage, obs)
                
                if target.id in comet_ids:
                    invalid_arrival = False
                    for group in obs.get('comets', []):
                        if target.id in group['planet_ids']:
                            if int(travel_time) >= len(group['paths'][0]) - group['path_index']:
                                invalid_arrival = True
                                break
                    if invalid_arrival: continue
                
                if self.is_path_safe(source.x, source.y, angle, math.hypot(tx - source.x, ty - source.y)):
                    mask[s_idx, t_idx] = True
        return mask

class ObservationProcessor:
    """
    Final audited ObservationProcessor for Orbit Wars.
    Ensures 100% alignment between raw list [id, owner, x, y, radius, ships, prod] 
    and Planet namedtuple structure.
    """
    def __init__(self, max_entities: int = 200, board_size: float = 100.0, max_speed: float = 6.0):
        self.max_entities = max_entities
        self.board_size = board_size
        self.max_speed = max_speed
        self.wrapper = OrbitWarsWrapper({"shipSpeed": max_speed, "boardSize": board_size})
        self.feature_dim = 18

    def process(self, obs: Dict[str, Any], player_id: int) -> Dict[str, np.ndarray]:
        entities = []
        entity_ids = []
        
        planets_raw = obs.get("planets", [])
        fleets_raw = obs.get("fleets", [])
        comet_ids = obs.get('comet_planet_ids', [])
        
        hub = None
        max_prod = -1
        for p_data in planets_raw:
            # Correct Mapping: [0:id, 1:owner, 2:x, 3:y, 4:radius, 5:ships, 6:production]
            if p_data[1] == player_id and p_data[6] > max_prod:
                max_prod = p_data[6]
                hub = Planet(*p_data[:7])
        
        for p_data in planets_raw:
            p_obj = Planet(*p_data[:7])
            feat = self._create_planet_features(p_obj, hub, obs, comet_ids)
            entities.append(feat)
            entity_ids.append(p_obj.id)

        for f_data in fleets_raw:
            f_obj = Fleet(*f_data)
            feat = self._create_fleet_features(f_obj, hub, obs)
            entities.append(feat)
            entity_ids.append(f_obj.id)

        num_entities = len(entities)
        if num_entities < self.max_entities:
            padding_size = self.max_entities - num_entities
            entities.extend([[0.0] * self.feature_dim] * padding_size)
            entity_ids.extend([0] * padding_size)
        else:
            entities = entities[:self.max_entities]
            entity_ids = entity_ids[:self.max_entities]

        return {
            "entities": np.array(entities, dtype=np.float32),
            "entity_ids": np.array(entity_ids, dtype=np.int64),
            "mask": np.array([1.0] * min(num_entities, self.max_entities) + [0.0] * max(0, self.max_entities - num_entities), dtype=np.float32)
        }

    def _create_planet_features(self, planet: Planet, hub: Planet, obs: Dict[str, Any], comet_ids: List[int]) -> List[float]:
        owner_oh = [0.0] * 5
        owner_oh[planet.owner + 1] = 1.0
        lin_ships = planet.ships / 1000.0
        log_ships = math.log(max(1, planet.ships)) / math.log(1000.0)
        hub_travel_time = 0.0
        hub_arrival_garrison = lin_ships
        if hub and hub.id != planet.id:
            planet_data = {'x': planet.x, 'y': planet.y, 'radius': planet.radius, 'id': planet.id, 'owner': planet.owner, 'production': planet.production, 'ships': planet.ships, 'source_ships': hub.ships}
            _, hub_travel_time, _, _ = self.wrapper.get_intercept_params((hub.x, hub.y), hub.radius, planet_data, 1.0, obs)
            raw_garrison = self.wrapper.estimate_future_garrison(planet_data, hub_travel_time)
            hub_arrival_garrison = raw_garrison / 1000.0
            hub_travel_time /= 100.0
        return [planet.id / 500.0, *owner_oh, planet.x / self.board_size, planet.y / self.board_size, planet.radius / 10.0, lin_ships, log_ships, planet.production / 5.0, 1.0 if planet.id in comet_ids else 0.0, 0.0, 0.0, 0.0, hub_travel_time, hub_arrival_garrison]

    def _create_fleet_features(self, fleet: Fleet, hub: Planet, obs: Dict[str, Any]) -> List[float]:
        owner_oh = [0.0] * 5
        owner_oh[fleet.owner + 1] = 1.0
        lin_ships = fleet.ships / 1000.0
        log_ships = math.log(max(1, fleet.ships)) / math.log(1000.0)
        speed = self.wrapper.calculate_speed(fleet.ships)
        # Using self.max_speed directly here, assume 6.0
        max_speed = 6.0
        vx = math.cos(fleet.angle) * speed / max_speed
        vy = math.sin(fleet.angle) * speed / max_speed
        hub_travel_time = 0.0
        if hub:
            dist = math.hypot(fleet.x - hub.x, fleet.y - hub.y)
            hub_speed = self.wrapper.calculate_speed(hub.ships)
            hub_travel_time = dist / hub_speed / 100.0
        return [fleet.id / 500.0, *owner_oh, fleet.x / self.board_size, fleet.y / self.board_size, 0.05, lin_ships, log_ships, 0.0, 0.0, 1.0, vx, vy, hub_travel_time, lin_ships]

class ActionProcessor:
    """
    Refactored ActionProcessor for Orbit Wars.
    Correctly maps Planet attributes [id, owner, x, y, radius, ships, production].
    """
    def __init__(self, wrapper: OrbitWarsWrapper):
        self.wrapper = wrapper

    def process_actions(self, obs: Dict[str, Any], player_id: int, 
                        target_indices: List[int], allocation_indices: List[int]) -> List[List[Any]]:
        all_moves = []
        planets_raw = obs.get("planets", [])
        # Empirical test confirms Planet(*p[:7]) is CORRECT for [id, owner, x, y, radius, ships, prod]
        planets = [Planet(*p[:7]) for p in planets_raw]
        
        for i, source in enumerate(planets):
            if source.owner != player_id:
                continue
            
            if i >= len(target_indices): break
            
            target_idx = target_indices[i]
            alloc_idx = allocation_indices[i]
            
            # FIX: Skip if target is the source itself (self-targeting is a "do-nothing" action)
            # or if the target index is invalid (padding/fleets).
            if alloc_idx == 0 or target_idx >= len(planets) or target_idx == i:
                continue
            
            target = planets[target_idx]
            target_data = {'x': target.x, 'y': target.y, 'radius': target.radius, 'id': target.id, 'owner': target.owner, 'production': target.production, 'ships': target.ships, 'source_ships': source.ships}
            
            if alloc_idx == 5:
                num_ships = target.ships + 5
                allocation_pct = 1.0
                for _ in range(5):
                    allocation_pct = min(1.0, num_ships / source.ships) if source.ships > 0 else 0
                    _, travel_time, _, _ = self.wrapper.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_pct, obs)
                    future_garrison = self.wrapper.estimate_future_garrison(target_data, travel_time)
                    new_num_ships = min(source.ships, future_garrison + 5)
                    if abs(new_num_ships - num_ships) < 1: break
                    num_ships = new_num_ships
            else:
                allocs = [0.0, 0.25, 0.5, 0.75, 1.0]
                allocation_pct = allocs[alloc_idx]
                num_ships = int(source.ships * allocation_pct)

            if num_ships <= 0: continue
            
            angle, travel_time, tx, ty = self.wrapper.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_pct, obs)
            dist_to_intercept = math.hypot(tx - source.x, ty - source.y)
            if self.wrapper.is_path_safe(source.x, source.y, angle, dist_to_intercept):
                all_moves.append([source.id, angle, num_ships])
                
        return all_moves
