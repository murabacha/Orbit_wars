# Orbit Wars Research Summary

This document summarizes the strategic and architectural findings for developing a competitive Orbit Wars agent, based on comprehensive research into RL paradigms and environment-specific dynamics.

## 1. Architectural Recommendations
- **Primary Algorithm:** **PPO (Proximal Policy Optimization)** is the recommended foundation due to its stability in multi-agent continuous-state environments.
- **Backbone:** **Transformer-Based RL**. Entities (planets, fleets, comets) should be treated as tokens. Self-attention allows the agent to weigh strategic threats globally without relying on fixed-size receptive fields.
- **Throughput:** **IMPALA** is suggested for massive distributed training (League training) but PPO is the safer baseline.

## 2. State & Action Representations
- **State Representation:** 
    - Tokenized entity list: `[N_entities, Features]`.
    - Features must include entity IDs to allow the network to learn "Geometric Tunneling" engine quirks.
    - Explicit travel time features to all targets to facilitate future garrison calculations.
- **Action Representation:**
    - **Hierarchical Decoupling:** The agent outputs high-level intent, not raw angles.
    - **Targeting:** Discrete selection of a Target Entity ID.
    - **Allocation:** Discrete ship allocation percentages: `[0%, 25%, 50%, 75%, 100%, exact_amount_needed]`.
    - **Wrapper:** A deterministic wrapper translates intent into firing angles using an 8-iteration mathematical intercept solver.

## 3. Reward Strategies
- **Primary Objective:** Sparse win/loss reward (+1/-1) at turn 500.
- **Dense Shaping (Early Learning):**
    - Potential-based rewards for ship production and planet capture.
    - Heavy incentives for early comet interaction to overcome interception difficulty.
    - Penalties for sun collisions and "Black Hole" comet deletions.

## 4. Training Methods
- **Initialization:** **Behavioral Cloning (BC)** of top-tier heuristic agents (e.g., Pilkwang) to learn latent physics and economics.
- **RL Phase:** PPO fine-tuning starting from BC weights.
- **Competitive Stability:** **League Training / Self-Play** with an opponent pool of historical checkpoints to prevent strategy unlearning.

## 5. Risks & Failure Modes
| Risk | Cause | Mitigation |
| :--- | :--- | :--- |
| **Geometric Tunneling** | List-order collision bias in engine. | Pass internal IDs as observable features. |
| **Black Hole Comets** | Race condition on comet expiration. | Wrapper must prohibit arrival on expiration turn via masking. |
| **Turtling Trap** | Fear of exposure leads to no-op. | Aggressive reward shaping and 4-player FFA training. |
| **Credit Assignment** | 500-turn delay for win signal. | Hierarchical RL or dense potential-based rewards. |

## 6. Design Implications
- The agent must calculate target garrisons at **projected time of arrival**.
- The inference loop must stay under 1.0s, precluding heavy online MCTS.
- Action masking is mandatory to prevent gradients from being polluted by invalid/physics-defying actions.
