import numpy as np
from typing import Dict, Any, List, Tuple
import math

class RewardShaper:
    """
    Pure Economic and Military Advantage Reward Shaper.
    Reduces micro-management to prevent agent passivity.
    """
    def __init__(self, player_id: int, gamma: float = 0.99, 
                 total_training_steps: int = 50_000_000):
        self.player_id = player_id
        self.gamma = gamma
        self.total_training_steps = total_training_steps
        
        # State tracking flags for potential evaluations
        self.is_initialized = False
        self.prev_raw_potential = 0.0
        self.last_dense_weight = 1.0

        # Global scale to keep rewards in a sane range for the Critic
        self.GLOBAL_REWARD_SCALE = 1000.0 

    def compute_raw_potential(self, obs: Dict[str, Any]) -> float:
        """
        Pure Economic and Military Advantage Potential.
        The agent must LEARN to batch ships because trickling results in dead ships (negative reward).
        """
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        
        # 1. Total Military Strength
        my_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
        enemy_ships = sum(p[5] for p in planets if p[1] not in [self.player_id, -1]) + sum(f[4] for f in fleets if f[1] not in [self.player_id, -1])
        
        # 2. Total Economic Engine
        my_production = sum(p[6] for p in planets if p[1] == self.player_id)
        enemy_production = sum(p[6] for p in planets if p[1] not in [self.player_id, -1])
        
        # Strategic Coefficients
        # Objective: Maximize the gap between us and the enemy.
        ship_advantage = (my_ships - enemy_ships) * 0.1
        prod_advantage = (my_production - enemy_production) * 5.0
        
        raw_potential = ship_advantage + prod_advantage
        return raw_potential

    def calculate_reward(self, obs: Dict[str, Any], done: bool, current_global_step: int = 0) -> float:
        """
        Calculates the telemetrically sound PBRS shaped reward term.
        Formula: Reward = Sparse_Signal + [gamma * Phi(s') - Phi(s)]
        """
        # Calculate active curriculum weight (annealing linearly from 1.0 to 0.0)
        dense_weight = max(0.0, 1.0 - (current_global_step / self.total_training_steps))
        self.last_dense_weight = dense_weight
        
        current_raw_potential = self.compute_raw_potential(obs)
        
        # FIX: If the episode is over, there is no future state. Potential must be 0.
        if done:
            current_raw_potential = 0.0
            
        if not self.is_initialized:
            self.prev_raw_potential = current_raw_potential
            self.is_initialized = True
            
        # Potential-Based Shaping Calculation: F = Phi(s') - Phi(s)
        # Using a simplified difference to keep gradients clean.
        shaped_reward = (current_raw_potential - self.prev_raw_potential) * dense_weight
        
        # Sparse Terminal Win/Loss Target Alignment
        terminal_reward = 0.0
        if done:
            planets = obs.get("planets", [])
            fleets = obs.get("fleets", [])
            my_ships = sum(p[5] for p in planets if p[1] == self.player_id) + sum(f[4] for f in fleets if f[1] == self.player_id)
            enemy_ships = sum(p[5] for p in planets if p[1] != self.player_id and p[1] != -1) + sum(f[4] for f in fleets if f[1] != self.player_id and f[1] != -1)

            # Massive spike to overcome potential collapse
            if my_ships > enemy_ships:
                terminal_reward = 5000.0  
            else:
                terminal_reward = -5000.0 
                
        # Aggregate and scale final signal
        total_reward = (terminal_reward + shaped_reward) / self.GLOBAL_REWARD_SCALE
        
        # Maintain reference state anchors
        if done:
            self.is_initialized = False
            self.prev_raw_potential = 0.0
        else:
            self.prev_raw_potential = current_raw_potential
            
        return total_reward
