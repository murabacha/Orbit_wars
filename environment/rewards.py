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
        self.prev_my_planets = 0.0
        self.last_dense_weight = 1.0

        # Global scale to keep rewards in a sane range for the Critic
        self.GLOBAL_REWARD_SCALE = 1000.0 

    def calculate_reward(self, obs: Dict[str, Any], done: bool, current_global_step: int = 0, episode_step: int = 0) -> float:
        # We don't strictly need dense_weight anymore, but keep it for structure
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        
        # Track both production AND raw planet count
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
            self.is_initialized = True

        # 1. Base capture reward (+50)
        # This makes capturing cheap neutral planets the fastest way to get points early.
        planet_capture_reward = (my_planets_now - getattr(self, 'prev_my_planets', my_planets_now)) * 50.0
        
        # 2. Production Reward (Stealing an enemy planet swings this massively)
        prod_reward = (my_prod_now - self.prev_my_prod) * 100.0 
        prod_penalty = (enemy_prod_now - self.prev_enemy_prod) * -100.0
        
        # 3. Time penalty (Forces speed. It loses points every turn it doesn't own the map)
        time_penalty = -0.50 
        
        # 4. The Kill Shot (Massive reward for 100% annihilation)
        terminal_reward = 0.0
        if done:
            if enemy_planets_now == 0:
                # 1000 base + huge bonus for finishing the game fast
                terminal_reward = 1000.0 + (500 - episode_step) * 2 
            elif my_total_ships > enemy_total_ships:
                terminal_reward = 100.0  
            else:
                terminal_reward = -500.0 
                
        step_reward = planet_capture_reward + prod_reward + prod_penalty + time_penalty + terminal_reward
        total_reward = step_reward / self.GLOBAL_REWARD_SCALE
        
        # Update anchors
        if done:
            self.is_initialized = False
        else:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            self.prev_my_planets = my_planets_now
            
        return total_reward
