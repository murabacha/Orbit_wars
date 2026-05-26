from typing import List, Dict, Any
from .wrapper import OrbitWarsWrapper
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

class ActionProcessor:
    """
    Translates RL Agent discrete intents into raw Kaggle environment moves.
    Implements the logic to select source planets and apply the intercept solver.
    """
    def __init__(self, wrapper: OrbitWarsWrapper):
        self.wrapper = wrapper

    def process_actions(self, obs: Dict[str, Any], player_id: int, 
                        target_indices: List[int], allocation_indices: List[int]) -> List[List[Any]]:
        """
        target_indices: Indices of entities in the observation's planet list.
        allocation_indices: Indices for [0%, 25%, 50%, 75%, 100%, exact_needed].
        """
        all_moves = []
        planets = [Planet(*p) for p in obs.get("planets", [])]
        my_planets = [p for p in planets if p.owner == player_id]
        
        # In this implementation, each of 'my_planets' is assigned a target/allocation 
        # from the agent's parallel output heads.
        for i, source in enumerate(my_planets):
            if i >= len(target_indices): break
            
            target_idx = target_indices[i]
            alloc_idx = allocation_indices[i]
            
            if alloc_idx == 0: continue # No action
            
            if target_idx >= len(planets): continue # Invalid target index (padding)
            target = planets[target_idx]
            
            # Allocation Logic
            allocs = [0.0, 0.25, 0.5, 0.75, 1.0]
            if alloc_idx < 5:
                num_ships = int(source.ships * allocs[alloc_idx])
            else:
                # Calculate exact needed + safety buffer
                _, travel_time, _, _ = self.wrapper.get_intercept_params(source, target, source.ships, obs.get('angular_velocity', 0))
                future_garrison = self.wrapper.estimate_future_garrison(target, travel_time)
                num_ships = min(source.ships, future_garrison + 5)
            
            if num_ships <= 0: continue
            
            angle, _, _, _ = self.wrapper.get_intercept_params(source, target, num_ships, obs.get('angular_velocity', 0))
            
            # Final Safety Check (Sun)
            dist = 10.0 # Just enough to clear the planet radius
            if self.wrapper.is_path_safe(source.x, source.y, angle, 100.0):
                all_moves.append([source.id, angle, num_ships])
                
        return all_moves
