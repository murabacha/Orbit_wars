import numpy as np
from typing import Dict, Any

class RewardShaper:
    """
    Stabilized RewardShaper for Orbit Wars.
    Fixes:
    1. Critic Explosion: Global scale to keep rewards in [-10, 10] range.
    2. Gamma Bleed: Removes gamma from potential difference to stop the living penalty.
    3. Curriculum Bleed: Applies dense_weight to the difference, not the absolute potential.
    """
    def __init__(self, player_id: int, gamma: float = 0.99, total_training_steps: int = 1_000_000):
        self.player_id = player_id
        self.gamma = gamma
        self.total_training_steps = total_training_steps
        
        self.is_initialized = False
        self.prev_raw_potential = 0.0
        self.last_dense_weight = 1.0

        # Protects the shared Transformer weights from gradient explosions
        self.GLOBAL_REWARD_SCALE = 1000.0 

    def compute_raw_potential(self, obs: Dict[str, Any]) -> float:
        """
        Computes the absolute potential Phi(s) of the active state.
        Uses strategic weights to encourage expansion and discourage over-fleet usage.
        """
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        comet_ids = obs.get('comet_planet_ids', [])
        
        my_planets = [p for p in planets if p[1] == self.player_id]
        my_fleets = [f for f in fleets if f[1] == self.player_id]
        
        # Correct index mapping: p[5]=ships, f[4]=ships
        ships_garrisoned = sum(p[5] for p in my_planets)
        ships_in_transit = sum(f[4] for f in my_fleets)
        total_production = sum(p[6] for p in my_planets)
        total_planets = len(my_planets)
        active_fleet_count = len(my_fleets)
        
        comet_bonus = sum(5.0 for p in my_planets if p[0] in comet_ids)
        
        # Strategic Coefficients
        w_ships_garrisoned = 0.15  
        w_ships_transit = 0.05     
        w_production = 5.0
        w_planets = 10.0
        w_fleet_penalty = -2.0     
        
        raw_potential = (
            (ships_garrisoned * w_ships_garrisoned) + 
            (ships_in_transit * w_ships_transit) + 
            (total_production * w_production) + 
            (total_planets * w_planets) + 
            (active_fleet_count * w_fleet_penalty) + 
            comet_bonus
        )
        return raw_potential

    def calculate_reward(self, obs: Dict[str, Any], done: bool, current_global_step: int = 0) -> float:
        """
        Calculates the refactored, stable reward signal.
        Formula: (Terminal_Signal + (Phi(s') - Phi(s)) * Dense_Weight) / 1000
        """
        # Calculate active curriculum weight (annealing linearly from 1.0 to 0.0)
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        current_raw_potential = self.compute_raw_potential(obs)
        
        # Terminate potential at episode end
        if done:
            current_raw_potential = 0.0
            
        if not self.is_initialized:
            self.prev_raw_potential = current_raw_potential
            self.is_initialized = True
            
        # FIX 1 & 2: Calculate difference FIRST to stop curriculum bleed, 
        # and remove gamma from the shaping to stop the 'living penalty' bleed.
        potential_diff = current_raw_potential - self.prev_raw_potential
        
        # Apply the annealing weight to the change in state, not the absolute state.
        shaped_reward = potential_diff * dense_weight
        
        # Terminal Win/Loss Target Alignment
        terminal_reward = 0.0
        if done:
            planets = obs.get("planets", [])
            fleets = obs.get("fleets", [])
            my_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
            enemy_ships = sum(p[5] for p in planets if p[1] != self.player_id and p[1] != -1) + sum(f[4] for f in fleets if f[1] != self.player_id and f[1] != -1)

            # design in thousands, but GLOBAL_REWARD_SCALE will keep it sane
            if my_ships > enemy_ships:
                terminal_reward = 5000.0 
            else:
                terminal_reward = -5000.0 
                
        # Aggregate unscaled signal
        total_unscaled_reward = terminal_reward + shaped_reward
        
        # Update state anchors
        if done:
            self.is_initialized = False
            self.prev_raw_potential = 0.0
        else:
            self.prev_raw_potential = current_raw_potential
            
        # FIX 3: Scale down the final reward to keep Value Loss MSE from exploding
        # This protects the shared Transformer features from massive gradients.
        return total_unscaled_reward / self.GLOBAL_REWARD_SCALE
