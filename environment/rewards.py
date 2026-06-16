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
        Only rewards capturing planets and penalizes time.
        """
        # Calculate active curriculum weight (annealing linearly from 1.0 to 0.0)
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        
        # 1. Calculate Production (Planet Ownership)
        my_prod_now = sum(p[6] for p in planets if p[1] == self.player_id)
        enemy_prod_now = sum(p[6] for p in planets if p[1] not in [self.player_id, -1])

        if not self.is_initialized:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            self.is_initialized = True

        # 2. MASSIVE reward ONLY for capturing planets (Production Delta)
        # It gets zero points for just sitting there letting ships grow.
        prod_reward = (my_prod_now - self.prev_my_prod) * 50.0 
        prod_penalty = (enemy_prod_now - self.prev_enemy_prod) * -50.0
        
        # 3. The "Hurry Up" Penalty
        # We charge the agent a tiny fee every step it leaves the enemy alive.
        # This prevents AFK farming and forces it to conquer the whole map.
        time_penalty = -0.05
        
        # 4. The Waypoint Dominance
        waypoint_bonus = 0.0
        # Trigger massive evaluations on Turns 50, 100, and 150
        if episode_step in [50, 100, 150]:
            my_total_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
            enemy_total_ships = sum(p[5] for p in planets if p[1] not in [self.player_id, -1]) + sum(f[4] for f in fleets if f[1] not in [self.player_id, -1])
            
            if my_total_ships > enemy_total_ships:
                waypoint_bonus = 50.0
            elif my_total_ships < enemy_total_ships:
                waypoint_bonus = -50.0

        # Sparse Terminal Win/Loss Target Alignment (Optional, keeping as a safety signal if done)
        terminal_reward = 0.0
        if done:
            my_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
            enemy_ships = sum(p[5] for p in planets if p[1] != self.player_id and p[1] != -1) + sum(f[4] for f in fleets if f[1] != self.player_id and f[1] != -1)
            if my_ships > enemy_ships:
                terminal_reward = 500.0  
            else:
                terminal_reward = -500.0 
                
        # Aggregate final signal
        # Note: We keep waypoint and terminal rewards separate from the production delta.
        step_reward = prod_reward + prod_penalty + time_penalty + waypoint_bonus + terminal_reward
        
        # Scale for stability
        total_reward = step_reward / self.GLOBAL_REWARD_SCALE
        
        # Maintain reference state anchors
        if done:
            self.is_initialized = False
            self.prev_my_prod = 0.0
            self.prev_enemy_prod = 0.0
        else:
            self.prev_my_prod = my_prod_now
            self.prev_enemy_prod = enemy_prod_now
            
        return total_reward
