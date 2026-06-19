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
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        
        # Track both production AND raw planet count to incentivize grabbing asteroids
        my_prod_now = sum(p[6] for p in planets if p[1] == self.player_id)
        enemy_prod_now = sum(p[6] for p in planets if p[1] not in [self.player_id, -1])
        my_planets_now = sum(1 for p in planets if p[1] == self.player_id)
        enemy_planets_now = sum(1 for p in planets if p[1] not in [self.player_id, -1])

        if not self.is_initialized:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            self.prev_my_planets = my_planets_now
            self.is_initialized = True

        # 1. EARLY EXPANSION: Base capture reward (+50 for any rock, even 0-prod ones)
        planet_capture_reward = (my_planets_now - getattr(self, 'prev_my_planets', my_planets_now)) * 50.0
        
        # 2. PRODUCTION SWING: Massive reward for stealing enemy bases
        prod_reward = (my_prod_now - self.prev_my_prod) * 100.0 
        prod_penalty = (enemy_prod_now - self.prev_enemy_prod) * -100.0
        
        # 3. SPEED ENFORCER: Bleed points every step to force fast play
        time_penalty = -0.50 
        
        # 4. HYPER-AGGRESSION: The Kill Shot vs The Cowardice Penalty
        terminal_reward = 0.0
        if done:
            if enemy_planets_now == 0:
                # 1000 base + massive speed bonus. Winning fast is the ONLY goal.
                terminal_reward = 1000.0 + (500 - episode_step) * 5 
            else:
                # NO CONSOLATION PRIZES. If timer runs out, penalize for surviving enemies.
                terminal_reward = -300.0 * enemy_planets_now 
                
        # Calculate active curriculum weight (annealing linearly from 1.0 to 0.0)
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        dense_reward = planet_capture_reward + prod_reward + prod_penalty + time_penalty
        step_reward = (dense_reward * dense_weight) + terminal_reward
        total_reward = step_reward / self.GLOBAL_REWARD_SCALE
        
        # Update anchors
        if done:
            self.is_initialized = False
        else:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            self.prev_my_planets = my_planets_now
            
        return total_reward
