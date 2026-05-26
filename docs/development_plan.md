# Development Plan: Orbit Wars AI

This plan outlines the roadmap for transitioning from basic heuristics to a superhuman Transformer-PPO agent.

## Phase 1: Foundation (Current)
- [x] Research synthesis and summary.
- [x] Environment wrappers (Intercept solver, Action masking).
- [x] Baseline Heuristic implementation.
- [x] Transformer-PPO and HRL architecture design.

## Phase 2: Behavioral Cloning
- **Goal:** Establish baseline latent representation.
- **Tasks:**
    - Generate 100k rollout samples from `HeuristicBaseline`.
    - Train `TransformerPPOModel` actor heads on these samples.
    - Validate that BC agent matches heuristic performance.

## Phase 3: RL Fine-Tuning
- **Goal:** Surpass heuristics.
- **Tasks:**
    - Initialize PPO with BC weights.
    - Train against fixed heuristics.
    - Implement dense reward shaping (Comet focus, Expansion rate).

## Phase 4: Competitive Robustness (League)
- **Goal:** Meta-game stability.
- **Tasks:**
    - Enable self-play with `SelfPlayManager`.
    - Curriculum: Gradually remove dense rewards in favor of sparse win/loss.
    - Stress-test against "Geometric Tunneling" and "Turtling" strategies.

## Phase 5: Deployment
- **Goal:** Kaggle submission.
- **Tasks:**
    - Optimize inference speed for < 1.0s act timeout.
    - Bundle all components into `main.py`.
