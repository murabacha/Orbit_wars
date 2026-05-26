# Training Specification: Transformer-PPO

## Framework
- **PyTorch** with **CleanRL** or **Ray RLlib** for PPO implementation.

## Hardware Requirements
- **Recommended:** 1x NVIDIA RTX 3090/4090 or A100 (for large batch sizes).
- **CPU:** 16+ Cores for parallel environment rollouts.

## Training Stages
1. **Stage 1: Behavioral Cloning (BC)**
    - Train on 50,000 trajectories from the `Baseline Heuristic`.
    - Purpose: Learn latent physics and basic expansion logic.
2. **Stage 2: PPO vs Heuristics**
    - Train against 3 static rule-based agents.
    - Purpose: Discover strategies that exploit common heuristic flaws.
3. **Stage 3: League Self-Play**
    - FFA self-play against historical checkpoints.
    - Purpose: Achieve strategic robustness and meta-stability.

## Hyperparameters
| Parameter | Value |
| :--- | :--- |
| Learning Rate | 3e-4 |
| Batch Size | 16384 (transitions) |
| Mini-batch Size | 2048 |
| Gamma | 0.998 |
| Entropy Coef | 0.01 |
| Max Grad Norm | 0.5 |

## Stopping Criteria
- Average Win Rate vs Top Heuristic > 85%.
- Policy entropy drops below 0.1 (signifying convergence).

## Expected Learning Behavior
- **0-1M steps:** Learning to avoid the sun and target neighbors.
- **1M-5M steps:** Perfecting comet interception and future garrison estimation.
- **5M+ steps:** Developing timed swarms and opportunistic third-party attacks.

## Troubleshooting
- **Sun Collisions:** Increase the negative reward penalty for fleet death.
- **Turtling:** Increase the `entropy_coef` or add a time-based penalty for no-ops.
