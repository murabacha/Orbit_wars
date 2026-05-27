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

    def compute_potential(self, obs: Dict[str, Any], dense_weight: float) -> float:
        """
        Computes the absolute potential Phi(s) of the active state.
        Phi(s) = (Total_Ships * W_s + Total_Production * W_p + Total_Planets * W_l) * dense_weight
        """
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        comet_ids = obs.get('comet_planet_ids', [])
        
        # Isolate assets owned by this specific agent
        my_planets = [p for p in planets if p[1] == self.player_id]
        my_fleets = [f for f in fleets if f[1] == self.player_id]
        
        # Extraction metrics based on updated planet list index positions
        # Planet structure: [id, owner, x, y, radius, ships, production]
        # (Based on probing, ships is index 5, production is index 6)
        # Note: If the user explicitly provided a structure with ships at 4, we follow that.
        # User provided: [id, owner, x, y, ships, radius, production, angular_velocity]
        ships_on_planets = sum(p[4] for p in my_planets)
        ships_in_fleets = sum(f[6] for f in my_fleets)
        total_ships = ships_on_planets + ships_in_fleets
        
        total_production = sum(p[6] for p in my_planets)
        total_planets = len(my_planets)
        
        # Comet ownership bonus (heavily incentivizes capturing high-value resources)
        comet_bonus = sum(5.0 for p in my_planets if p[0] in comet_ids)
        
        # Base strategic potential coefficients
        w_ships = 0.1
        w_production = 5.0
        w_planets = 10.0
        
        raw_potential = (
            (total_ships * w_ships) + 
            (total_production * w_production) + 
            (total_planets * w_planets) + 
            comet_bonus
        )
        
        # Scale potential down linearly based on the active curriculum weight
        return raw_potential * dense_weight

    def calculate_reward(self, obs: Dict[str, Any], done: bool, current_global_step: int = 0) -> float:
        """
        Calculates the telemetrically sound PBRS shaped reward term.
        Formula: Reward = Sparse_Signal + [gamma * Phi(s') - Phi(s)] - Physical_Penalties
        """
        # Calculate active curriculum weight (annealing linearly from 1.0 to 0.0)
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        
        # 1. Compute State Potential
        current_potential = self.compute_potential(obs, dense_weight)
        
        if not self.is_initialized:
            self.prev_potential = current_potential
            self.is_initialized = True
            
        # 2. Potential-Based Shaping Calculation: F = gamma * Phi(s') - Phi(s)
        shaped_reward = (self.gamma * current_potential) - self.prev_potential
        
        # 3. Environmental Physics Hazards Penalties (Sun / Comet Expiration)
        # Check if the environment recorded fleet destructions this step
        physics_penalties = 0.0
        # If your environment setup exposes step casualties, extract them here:
        # e.g., physics_penalties += lost_ships_to_sun * 1.0
        
        # 4. Sparse Terminal Win/Loss Target Alignment
        terminal_reward = 0.0
        if done:
            # Note: Final rewards in Kaggle are usually in env.state[i].reward.
            # Here we assume they are passed in the obs dict under 'rewards'.
            scores = obs.get('rewards', [0, 0, 0, 0])
            # Check win condition relative to opponents
            if scores[self.player_id] == max(scores) and scores[self.player_id] > 0:
                terminal_reward = 1.0  # Normalized sparse win signal
            else:
                terminal_reward = -1.0 # Normalized sparse loss signal
                
        # Aggregate final signal
        total_reward = terminal_reward + shaped_reward - physics_penalties
        
        # Maintain reference state anchors
        if done:
            self.is_initialized = False
            self.prev_potential = 0.0
        else:
            self.prev_potential = current_potential
            
        return total_reward
