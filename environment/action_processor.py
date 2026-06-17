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
            
            # 2. Iterative Multi-Step Convergence for Allocation 5 (Exact Needed)
            if alloc_idx == 5:
                # FIX: Don't run conquer math on our own planets to avoid wasting ships
                if target.owner == player_id:
                    num_ships = int(source.ships * 0.25)
                    allocation_pct = 0.25
                else:
                    # Resolve the circular dependency: num_ships -> speed -> travel_time -> future_garrison -> num_ships
                    num_ships = target.ships + 5
                    allocation_pct = 1.0
                    
                    for _ in range(5): # Convergence loop
                        allocation_pct = min(1.0, num_ships / source.ships) if source.ships > 0 else 0
                        _, travel_time, _, _ = self.wrapper.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_pct, obs)
                        future_garrison = self.wrapper.estimate_future_garrison(target_data, travel_time)
                        
                        new_num_ships = min(source.ships, future_garrison + 5)
                        if abs(new_num_ships - num_ships) < 1:
                            break
                        num_ships = new_num_ships
            else:
                # Standard Percentage Bins
                allocs = [0.0, 0.25, 0.5, 0.75, 1.0]
                allocation_pct = allocs[alloc_idx]
                num_ships = int(source.ships * allocation_pct)

            # THE BATCHING FIX: 
            # Drop the threshold to 10 for faster early games.
            # EXCEPTION: Allow <10 if it's an Exact strike (5) or a desperate 100% all-in (4).
            if num_ships < 10 and alloc_idx not in [4, 5]: 
                continue
                
            # THE SHUFFLING FIX:
            # Prevent moving ships to our own planets unless using the exact intercept (alloc 5)
            # This forces the network to spend its actions on outward aggression.
            if target.owner == player_id and alloc_idx != 5:
                continue
            
            # 3. Final Intercept Calculation
            angle, travel_time, tx, ty = self.wrapper.get_intercept_params((source.x, source.y), source.radius, target_data, allocation_pct, obs)
            
            # 4. Precise Path-to-Sun Safety
            # Calculate safety using the distance to the moving intercept coordinate
            dist_to_intercept = math.hypot(tx - source.x, ty - source.y)
            if self.wrapper.is_path_safe(source.x, source.y, angle, dist_to_intercept):
                all_moves.append([source.id, angle, num_ships])
                
        return all_moves
