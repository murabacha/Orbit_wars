import numpy as np
from typing import Dict, Any, List, Tuple
import math

class RewardShaper:
    """
    "Total Annihilation" Reward Shaper.
    Focuses exclusively on production dominance and speed.
    """
    def __init__(self, player_id: int, gamma: float = 0.99, 
                 total_training_steps: int = 50_000_000):
        self.player_id = player_id
        self.gamma = gamma
        self.total_training_steps = total_training_steps
        
        # State tracking flags for evaluations
        self.is_initialized = False
        self.prev_my_prod = 0.0
        self.prev_enemy_prod = 0.0
        self.last_dense_weight = 1.0

        # Global scale to keep rewards in a sane range for the Critic
        self.GLOBAL_REWARD_SCALE = 1000.0 

    def calculate_reward(self, obs: Dict[str, Any], done: bool, current_global_step: int = 0, episode_step: int = 0) -> float:
        """
        Calculates the "Total Annihilation" reward.
        Includes base capture rewards, production delta, and dense ship advantage.
        """
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        
        # Track both production AND raw planet count to fix the asteroid issue
        my_prod_now = sum(p[6] for p in planets if p[1] == self.player_id)
        enemy_prod_now = sum(p[6] for p in planets if p[1] not in [self.player_id, -1])
        my_planets_now = sum(1 for p in planets if p[1] == self.player_id)
        enemy_planets_now = sum(1 for p in planets if p[1] not in [self.player_id, -1])
        
        my_total_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
        enemy_total_ships = sum(p[5] for p in planets if p[1] not in [self.player_id, -1]) + sum(f[4] for f in fleets if f[1] not in [self.player_id, -1])

        if not self.is_initialized:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            self.prev_my_planets = my_planets_now
            self.prev_my_ships = my_total_ships
            self.prev_enemy_ships = enemy_total_ships
            self.is_initialized = True

        # 1. Base capture reward (Fixes Problem 3: now it values 0-production asteroids)
        planet_capture_reward = (my_planets_now - getattr(self, 'prev_my_planets', my_planets_now)) * 20.0
        
        # 2. Production Reward
        prod_reward = (my_prod_now - self.prev_my_prod) * 50.0 
        prod_penalty = (enemy_prod_now - self.prev_enemy_prod) * -50.0
        
        # 3. Dense Ship Advantage (Fixes Problem 5: rewards favorable trades and widening the gap)
        ship_advantage_now = my_total_ships - enemy_total_ships
        prev_ship_advantage = self.prev_my_ships - self.prev_enemy_ships
        advantage_reward = (ship_advantage_now - prev_ship_advantage) * 0.1 * dense_weight
        
        # 4. The "Hurry Up" Penalty
        time_penalty = -0.05
        
        # 5. Terminal Win/Loss (Fixes Problem 2 & 4: aggressively rewards wiping the enemy out)
        terminal_reward = 0.0
        if done:
            if enemy_planets_now == 0:
                # Massive bonus for an actual knockout 
                terminal_reward = 1000.0 + (500 - episode_step) # Bonus for doing it fast
            elif my_total_ships > enemy_total_ships:
                # Small consolation prize for a timeout win, prevents hoarding
                terminal_reward = 100.0  
            else:
                terminal_reward = -500.0 
                
        step_reward = planet_capture_reward + prod_reward + prod_penalty + advantage_reward + time_penalty + terminal_reward
        total_reward = step_reward / self.GLOBAL_REWARD_SCALE
        
        # Update anchors
        if done:
            self.is_initialized = False
        else:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            self.prev_my_planets = my_planets_now
            self.prev_my_ships = my_total_ships
            self.prev_enemy_ships = enemy_total_ships
            
        return total_reward
