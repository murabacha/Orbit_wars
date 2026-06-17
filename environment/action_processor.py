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
                        target_indices: List[int], allocation_indices: List[int]) -> List[List[Any]]:
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
            alloc_idx = allocation_indices[i]
            
            # FIX: Skip if target is the source itself (self-targeting is a "do-nothing" action)
            # or if the target index is invalid (padding/fleets).
            if alloc_idx == 0 or target_idx >= len(planets) or target_idx == i:
                continue
            
            target = planets[target_idx]
            target_data = {'x': target.x, 'y': target.y, 'radius': target.radius, 'id': target.id, 'owner': target.owner, 'production': target.production, 'ships': target.ships, 'source_ships': source.ships}
            
            # Standard Percentage Bins ONLY. Clamp anything above 4.
            allocs = [0.0, 0.25, 0.5, 0.75, 1.0]
            safe_alloc_idx = min(alloc_idx, 4) 
            allocation_pct = allocs[safe_alloc_idx]
            
            num_ships = int(source.ships * allocation_pct)

            # THE UNBREAKABLE WALL
            if num_ships < 15: 
                continue
                
            # THE SHUFFLING FIX:
            # Prevent moving ships to our own planets unless using 100% (evacuation)
            if target.owner == player_id and safe_alloc_idx != 4:
                continue
            
            # 3. Final Intercept Calculation
            angle, travel_time, tx, ty = self.wrapper.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_pct, obs)
            
            # 4. Precise Path-to-Sun Safety
            # Calculate safety using the distance to the moving intercept coordinate
            dist_to_intercept = math.hypot(tx - source.x, ty - source.y)
            if self.wrapper.is_path_safe(source.x, source.y, angle, dist_to_intercept):
                all_moves.append([source.id, angle, num_ships])
                
        return all_moves
