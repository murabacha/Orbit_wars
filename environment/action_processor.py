import math
from typing import List, Dict, Any, Tuple
from .wrapper import OrbitWarsWrapper
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

class ActionProcessor:
    """
    Refactored ActionProcessor for Orbit Wars.
    Correctly maps Planet attributes [id, owner, x, y, radius, ships, production].
    """
    def __init__(self, wrapper: OrbitWarsWrapper):
        self.wrapper = wrapper

    def process_actions(self, obs: Dict[str, Any], player_id: int, 
                        target_indices: List[int], allocation_indices: List[int],
                        min_ships: float = 15.0) -> List[List[Any]]:
        all_moves = []
        planets_raw = obs.get("planets", [])
        # Empirical test confirms Planet(*p[:7]) is CORRECT for [id, owner, x, y, radius, ships, prod]
        planets = [Planet(*p[:7]) for p in planets_raw]
        
        comet_map = {}
        for group in obs.get('comets', []):
            for p_id in group['planet_ids']:
                comet_map[p_id] = group
 
        for i, source in enumerate(planets):
            if source.owner != player_id:
                continue
            
            if i >= len(target_indices): break
            
            target_idx = target_indices[i]
            alloc_idx = min(max(0, allocation_indices[i]), 100)
            
            # FIX: Skip if target is the source itself (self-targeting is a "do-nothing" action)
            # or if the target index is invalid (padding/fleets).
            if alloc_idx == 0 or target_idx >= len(planets) or target_idx == i:
                continue
            
            target = planets[target_idx]
            target_data = {'x': target.x, 'y': target.y, 'radius': target.radius, 'id': target.id, 'owner': target.owner, 'production': target.production, 'ships': target.ships, 'source_ships': source.ships}
            
            # Allocation Bins: Hybrid Space
            # Bins 0-75: send exact absolute number of ships (0 to 75).
            # Bins 76-100: send predefined percentage (0% to 100% in 4.17% steps).
            if alloc_idx <= 75:
                num_ships = min(alloc_idx, source.ships)
                allocation_pct = (num_ships / float(source.ships)) if source.ships > 0 else 0.0
            else:
                allocation_pct = (alloc_idx - 76) / 24.0
                num_ships = int(source.ships * allocation_pct)
 
            # THE UNBREAKABLE WALL (with exception for attacks that can capture the target)
            is_attack = (target.owner != player_id)
            if num_ships < min_ships:
                if not (is_attack and num_ships > target.ships):
                    continue
                
            # THE SHUFFLING FIX:
            # Prevent moving ships to our own planets unless using 100% (evacuation, bin 100)
            if target.owner == player_id and alloc_idx != 100:
                continue
            
            # 3. Final Intercept Calculation
            angle, travel_time, tx, ty = self.wrapper.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_pct, obs)
            
            # 4. Precise Path-to-Sun Safety
            # Calculate safety using the distance to the moving intercept coordinate
            dist_to_intercept = math.hypot(tx - source.x, ty - source.y)
            if self.wrapper.is_path_safe(source.x, source.y, angle, dist_to_intercept):
                all_moves.append([source.id, angle, num_ships])
                
        return all_moves
