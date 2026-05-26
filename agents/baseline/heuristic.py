import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

class HeuristicBaseline:
    """
    An advanced rule-based agent used for Behavioral Cloning and as a training opponent.
    Prioritizes comets and calculates future garrisons.
    """
    def __init__(self, player_id: int):
        self.player_id = player_id

    def act(self, obs: dict) -> list:
        moves = []
        planets = [Planet(*p) for p in obs.get("planets", [])]
        my_planets = [p for p in planets if p.owner == self.player_id]
        targets = [p for p in planets if p.owner != self.player_id]
        
        if not my_planets or not targets:
            return []

        # 1. Prioritize Comets
        comet_ids = obs.get('comet_planet_ids', [])
        comet_targets = [p for p in targets if p.id in comet_ids]
        
        # 2. Simple Expansion Logic
        for source in my_planets:
            if source.ships < 5: continue
            
            # Find best target (High production / Low cost)
            best_target = None
            best_score = -1e9
            
            potential_targets = comet_targets if comet_targets else targets
            
            for t in potential_targets:
                dist = math.hypot(source.x - t.x, source.y - t.y)
                # Heuristic Score: Prod / (Dist * Ships)
                score = (t.production * 10) / (dist * (t.ships + 1))
                if score > best_score:
                    best_score = score
                    best_target = t
            
            if best_target:
                # Intercept Math (Simplified)
                angle = math.atan2(best_target.y - source.y, best_target.x - source.x)
                # Send 70% of ships or enough to cap
                ships_to_send = min(source.ships - 1, int(best_target.ships * 1.5) + 2)
                if ships_to_send > 0:
                    moves.append([source.id, angle, ships_to_send])
                    
        return moves
