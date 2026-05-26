# Experiment Plan

## Experiment 1: BC vs Pure RL
- **Objective:** Compare convergence speed of an agent initialized with Behavioral Cloning vs an agent starting from random weights.
- **Hypothesis:** BC agent will reach heuristic performance 10x faster and avoid the "Sun Collision" local optimum.

## Experiment 2: Intercept Iteration Count
- **Objective:** Determine the impact of intercept solver precision (1 vs 4 vs 8 iterations).
- **Hypothesis:** 8 iterations are necessary for capturing fast-moving comets on elliptical paths.

## Experiment 3: Reward Shaping Weighting
- **Objective:** Test sensitivity to Planet Capture vs Ship Production rewards.
- **Success Criteria:** Highest final ship count in self-play.
