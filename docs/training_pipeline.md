# Training Pipeline

## 1. Experience Collection
- **Environment:** Custom Orbit Wars Wrapper.
- **Actors:** Parallel processes running rollouts.
- **Buffer:** Stores transitions `(s, a, r, s', log_prob, mask)`.

## 2. PPO Update Cycle
- **Advantage Calculation:** Uses GAE (Generalized Advantage Estimation) with $\lambda=0.95$.
- **Policy Loss:** Clipped surrogate objective.
- **Value Loss:** Mean Squared Error (MSE) of state value prediction.
- **Entropy Bonus:** Prevents premature convergence.

## 3. League Strategy
To prevent "strategy cycling" (where A beats B, B beats C, but C beats A), the system uses **Prioritized Fictitious Self-Play**.
- 80% of matches are against the latest agent.
- 20% are against older, diverse checkpoints.
