import numpy as np
import math
from typing import Dict, Any, List
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from .wrapper import OrbitWarsWrapper

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
        
        # --- NEW: Visual Memory ---
        fleets = obs.get("fleets", [])
        incoming_friendly = sum(f[4] for f in fleets if f[1] == planet.owner and f[1] != -1 and f[3] == planet.id)
        incoming_enemy = sum(f[4] for f in fleets if f[1] != planet.owner and f[1] != -1 and f[3] == planet.id)
        # --------------------------

        hub_travel_time = 0.0
        hub_arrival_garrison = lin_ships
        if hub and hub.id != planet.id:
            planet_data = {'x': planet.x, 'y': planet.y, 'radius': planet.radius, 'id': planet.id, 'owner': planet.owner, 'production': planet.production, 'ships': planet.ships, 'source_ships': hub.ships}
            _, hub_travel_time, _, _ = self.wrapper.get_intercept_params((hub.x, hub.y), hub.radius, planet_data, 1.0, obs)
            raw_garrison = self.wrapper.estimate_future_garrison(planet_data, hub_travel_time)
            hub_arrival_garrison = raw_garrison / 1000.0
            hub_travel_time /= 100.0
            
        # Replaced the 0.0 padding slots with our incoming ship features!
        return [planet.id / 500.0, *owner_oh, planet.x / self.board_size, planet.y / self.board_size, planet.radius / 10.0, 
                lin_ships, log_ships, planet.production / 5.0, 1.0 if planet.id in comet_ids else 0.0, 
                incoming_friendly / 1000.0, incoming_enemy / 1000.0, 0.0, hub_travel_time, hub_arrival_garrison]

    def _create_fleet_features(self, fleet: Fleet, hub: Planet, obs: Dict[str, Any]) -> List[float]:
        owner_oh = [0.0] * 5
        owner_oh[fleet.owner + 1] = 1.0
        lin_ships = fleet.ships / 1000.0
        log_ships = math.log(max(1, fleet.ships)) / math.log(1000.0)
        speed = self.wrapper.calculate_speed(fleet.ships)
        vx = math.cos(fleet.angle) * speed / self.max_speed
        vy = math.sin(fleet.angle) * speed / self.max_speed
        hub_travel_time = 0.0
        if hub:
            dist = math.hypot(fleet.x - hub.x, fleet.y - hub.y)
            hub_speed = self.wrapper.calculate_speed(hub.ships)
            hub_travel_time = dist / hub_speed / 100.0
        return [fleet.id / 500.0, *owner_oh, fleet.x / self.board_size, fleet.y / self.board_size, 0.05, lin_ships, log_ships, 0.0, 0.0, 1.0, vx, vy, hub_travel_time, lin_ships]
