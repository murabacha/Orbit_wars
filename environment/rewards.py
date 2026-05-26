from typing import Dict, Any

class RewardShaper:
    """
    Implements dense reward shaping to guide RL agents through the sparse Orbit Wars environment.
    """
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.prev_ships = 0
        self.prev_planets = 0
        self.prev_production = 0

    def calculate_reward(self, obs: Dict[str, Any], done: bool) -> float:
        player_id = self.player_id
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        
        # Current Stats
        my_planets = [p for p in planets if p[1] == player_id]
        my_fleets = [f for f in fleets if f[1] == player_id]
        
        current_ships = sum(p[5] for p in my_planets) + sum(f[6] for f in my_fleets)
        current_planets = len(my_planets)
        current_production = sum(p[6] for p in my_planets)
        
        reward = 0.0
        
        # 1. Delta Production (Economic growth)
        reward += (current_production - self.prev_production) * 5.0
        
        # 2. Delta Planets (Expansion)
        reward += (current_planets - self.prev_planets) * 10.0
        
        # 3. Comet Interaction (High value bonus)
        comet_ids = obs.get('comet_planet_ids', [])
        for p in my_planets:
            if p[0] in comet_ids:
                reward += 2.0 # Passive reward for holding a comet
        
        # 4. Survival / Ship Count delta
        reward += (current_ships - self.prev_ships) * 0.1
        
        # 5. Sparse Win/Loss terminal reward
        if done:
            # Win = 100, Loss = -100
            # Note: This should be normalized/calibrated with dense rewards
            scores = obs.get('rewards', [0, 0, 0, 0])
            if scores[player_id] == max(scores) and scores[player_id] > 0:
                reward += 100.0
            else:
                reward -= 100.0

        # Update state
        self.prev_ships = current_ships
        self.prev_planets = current_planets
        self.prev_production = current_production
        
        return reward
