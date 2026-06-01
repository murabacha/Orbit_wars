import numpy as np
import math
from typing import Dict, Any, List
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from .wrapper import OrbitWarsWrapper

class ObservationProcessor:
    """
    Refactored ObservationProcessor for Orbit Wars.
    Features:
    - True simulation IDs (p_id, f_id).
    - Linear and Logarithmic ship metrics.
    - True directional flight vectors (vx, vy) for fleets.
    - Predictive features: travel time and arrival garrison from primary hub.
    """
    def __init__(self, max_entities: int = 200, board_size: float = 100.0, max_speed: float = 6.0):
        self.max_entities = max_entities
        self.board_size = board_size
        self.max_speed = max_speed
        self.wrapper = OrbitWarsWrapper({"shipSpeed": max_speed, "boardSize": board_size})
        # Feature mapping:
        # [Norm_ID, Owner_OH(5), Norm_X, Norm_Y, Norm_R, 
        #  Lin_Ships, Log_Ships, Norm_Prod, 
        #  Is_Comet, Is_Fleet, VX, VY, 
        #  Hub_Travel_Time, Hub_Arrival_Garrison]
        self.feature_dim = 1 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 # Total 18 features

    def process(self, obs: Dict[str, Any], player_id: int) -> Dict[str, np.ndarray]:
        entities = []
        entity_ids = []
        
        planets_raw = obs.get("planets", [])
        fleets_raw = obs.get("fleets", [])
        comet_ids = obs.get('comet_planet_ids', [])
        
        # 1. Identify Primary Hub (Highest production planet owned by player)
        hub = None
        max_prod = -1
        for p_data in planets_raw:
            # Updated structure: [id, owner, x, y, ships, radius, production, ...]
            if p_data[1] == player_id and p_data[6] > max_prod:
                max_prod = p_data[6]
                # Map to standard Planet object for consistency if needed, 
                # but ensure we use the correct index for ships/radius
                hub = Planet(p_data[0], p_data[1], p_data[2], p_data[3], p_data[5], p_data[4], p_data[6])
        
        # 2. Process Planets
        for p_data in planets_raw:
            p_obj = Planet(p_data[0], p_data[1], p_data[2], p_data[3], p_data[5], p_data[4], p_data[6])
            feat = self._create_planet_features(p_obj, hub, obs, comet_ids)
            entities.append(feat)
            entity_ids.append(p_obj.id)

        # 3. Process Fleets
        for f_data in fleets_raw:
            f_obj = Fleet(*f_data)
            feat = self._create_fleet_features(f_obj, hub, obs)
            entities.append(feat)
            entity_ids.append(f_obj.id)

        # 4. Padding
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
        # Basic Features
        owner_oh = [0.0] * 5
        owner_oh[planet.owner + 1] = 1.0
        
        # Ship Metrics
        lin_ships = planet.ships / 1000.0
        log_ships = math.log(max(1, planet.ships)) / math.log(1000.0)
        
        # Predictive Features from Hub
        hub_travel_time = 0.0
        hub_arrival_garrison = lin_ships
        if hub and hub.id != planet.id:
            planet_data = {
                'x': planet.x, 'y': planet.y, 'id': planet.id, 
                'owner': planet.owner, 'production': planet.production, 
                'ships': planet.ships, 'source_ships': hub.ships
            }
            _, hub_travel_time, _, _ = self.wrapper.get_intercept_params((hub.x, hub.y), planet_data, 1.0, obs)
            raw_garrison = self.wrapper.estimate_future_garrison(planet_data, hub_travel_time)
            hub_arrival_garrison = raw_garrison / 1000.0
            hub_travel_time /= 100.0 # Normalize turns
            
        return [
            planet.id / 500.0,
            *owner_oh,
            planet.x / self.board_size,
            planet.y / self.board_size,
            planet.radius / 10.0,
            lin_ships,
            log_ships,
            planet.production / 5.0,
            1.0 if planet.id in comet_ids else 0.0,
            0.0, # Is_Fleet
            0.0, # VX
            0.0, # VY
            hub_travel_time,
            hub_arrival_garrison
        ]

    def _create_fleet_features(self, fleet: Fleet, hub: Planet, obs: Dict[str, Any]) -> List[float]:
        # Basic Features
        owner_oh = [0.0] * 5
        owner_oh[fleet.owner + 1] = 1.0
        
        # Ship Metrics
        lin_ships = fleet.ships / 1000.0
        log_ships = math.log(max(1, fleet.ships)) / math.log(1000.0)
        
        # Velocity Vectors
        # In Orbit Wars, fleet.angle is raw continuous angle. Speed is dynamic.
        speed = self.wrapper.calculate_speed(fleet.ships)
        vx = math.cos(fleet.angle) * speed / self.max_speed
        vy = math.sin(fleet.angle) * speed / self.max_speed
        
        # Predictive Features from Hub (Travel time to moving fleet)
        hub_travel_time = 0.0
        if hub:
            dist = math.hypot(fleet.x - hub.x, fleet.y - hub.y)
            # Simple approximation for fleet intercept
            hub_speed = self.wrapper.calculate_speed(hub.ships)
            hub_travel_time = dist / hub_speed / 100.0
            
        return [
            fleet.id / 500.0,
            *owner_oh,
            fleet.x / self.board_size,
            fleet.y / self.board_size,
            0.05, # Fixed Norm_R for fleets
            lin_ships,
            log_ships,
            0.0,  # Production
            0.0,  # Is_Comet
            1.0,  # Is_Fleet
            vx,
            vy,
            hub_travel_time,
            lin_ships # Garrison is just current ships for moving fleet
        ]
