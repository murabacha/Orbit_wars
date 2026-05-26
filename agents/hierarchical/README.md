# Hierarchical RL Agent (HRL)

This agent uses temporal abstraction to solve the 500-turn credit assignment problem.

## Architecture
- **Manager:** A neural network that selects a "Macro Mode" every 10 turns.
- **Worker:** A deterministic or sub-policy agent that executes the chosen mode.

## Macro Modes
1. **Expansion:** Prioritize capturing neutral planets with high production.
2. **Consolidation:** Reinforce vulnerable home planets and build garrisons.
3. **Aggression:** Target the weakest player or high-value enemy hubs.
4. **Comet Focus:** Divert all resources to intercepting spawning comets.

## Why HRL?
Orbit Wars is a long-duration game. By committing to a strategy for 10 turns, the Manager learns to value long-term economic growth over immediate tactical gains.
