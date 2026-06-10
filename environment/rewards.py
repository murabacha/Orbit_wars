import numpy as np
from typing import Dict, Any, List, Tuple
import math

class RewardShaper:
    """
    Implements mathematically rigorous Potential-Based Reward Shaping (PBRS) 
    for Orbit Wars. Ensures policy invariance while guiding early exploration.
    Includes a step-based curriculum schedule to anneal dense terms over training.
    """
    def __init__(self, player_id: int, gamma: float = 0.99, 
                 total_training_steps: int = 50_000_000):
        self.player_id = player_id
        self.gamma = gamma
        self.total_training_steps = total_training_steps
        
        # State tracking flags for potential evaluations
        self.is_initialized = False
        self.prev_potential = 0.0
        self.last_dense_weight = 1.0

    def compute_potential(self, obs: Dict[str, Any], dense_weight: float) -> float:
        """
        Computes the absolute potential Phi(s) of the active state.
        Phi(s) = (Garrisoned_Ships * W_g + Transit_Ships * W_t + Total_Production * W_p + Total_Planets * W_l - Fleet_Count * W_f) * dense_weight
        """
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        comet_ids = obs.get('comet_planet_ids', [])
        
        # Isolate assets owned by this specific agent
        my_planets = [p for p in planets if p[1] == self.player_id]
        my_fleets = [f for f in fleets if f[1] == self.player_id]
        
        # FIX 2: Split ships into garrisoned and transit to penalize long travel times
        # Planet structure: [id, owner, x, y, radius, ships, production]
        ships_garrisoned = sum(p[5] for p in my_planets)
        # Fleet structure: [id, owner, source, target, ships, current_steps, total_steps]
        ships_in_transit = sum(f[4] for f in my_fleets)
        
        total_production = sum(p[6] for p in my_planets)
        total_planets = len(my_planets)
        active_fleet_count = len(my_fleets)
        
        # Comet ownership bonus
        comet_bonus = sum(5.0 for p in my_planets if p[0] in comet_ids)
        
        # STRATEGIC WEIGHTS
        w_ships_garrisoned = 0.15  # High value for safe ships
        w_ships_transit = 0.05     # Transit Depreciation (encourages short flights)
        w_production = 5.0
        w_planets = 10.0
        w_fleet_penalty = -2.0     # The Fleet Tax (encourages batching)
        
        raw_potential = (
            (ships_garrisoned * w_ships_garrisoned) + 
            (ships_in_transit * w_ships_transit) + 
            (total_production * w_production) + 
            (total_planets * w_planets) + 
            (active_fleet_count * w_fleet_penalty) + 
            comet_bonus
        )
        
        # Scale potential down linearly based on the active curriculum weight
        return max(0.0, raw_potential * dense_weight)

    def calculate_reward(self, obs: Dict[str, Any], done: bool, current_global_step: int = 0) -> float:
        """
        Calculates the telemetrically sound PBRS shaped reward term.
        Formula: Reward = Sparse_Signal + [gamma * Phi(s') - Phi(s)] - Physical_Penalties
        """
        # Calculate active curriculum weight (annealing linearly from 1.0 to 0.0)
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        # 1. Compute State Potential
        current_potential = self.compute_potential(obs, dense_weight)
        
        # FIX: If the episode is over, there is no future state. Potential must be 0.
        if done:
            current_potential = 0.0
            
        if not self.is_initialized:
            self.prev_potential = current_potential
            self.is_initialized = True
            
        # 2. Potential-Based Shaping Calculation: F = gamma * Phi(s') - Phi(s)
        shaped_reward = (self.gamma * current_potential) - self.prev_potential
        
        # 3. Environmental Physics Hazards Penalties (Sun / Comet Expiration)
        physics_penalties = 0.0
        
        # 4. Sparse Terminal Win/Loss Target Alignment
        terminal_reward = 0.0
        if done:
            planets = obs.get("planets", [])
            fleets = obs.get("fleets", [])
            my_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
            enemy_ships = sum(p[5] for p in planets if p[1] != self.player_id and p[1] != -1) + sum(f[4] for f in fleets if f[1] != self.player_id and f[1] != -1)

            # FIX 3: Massive scale increase (±5000.0) to overcome PBRS drop-off dilution
            if my_ships > enemy_ships:
                terminal_reward = 5000.0  # Massive spike for victory
            else:
                terminal_reward = -5000.0 # Massive penalty for loss
                
        # Aggregate final signal
        total_reward = terminal_reward + shaped_reward - physics_penalties
        
        # Maintain reference state anchors
        if done:
            self.is_initialized = False
            self.prev_potential = 0.0
        else:
            self.prev_potential = current_potential
            
        return total_reward
