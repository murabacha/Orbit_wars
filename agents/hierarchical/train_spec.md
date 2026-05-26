# Training Specification: Hierarchical RL

## Framework
- **PyTorch** with custom HRL loop.

## Training Stages
1. **Stage 1: Rule-Based Worker Calibration**
    - Ensure the low-level workers (Expand, Defend, etc.) are robust.
2. **Stage 2: Manager Training**
    - Train the Manager to select modes that maximize the final ship count.
    - Uses a sparse reward delivered only at turn 500.

## Stopping Criteria
- Manager policy converges on a stable macro-priority (e.g., favoring expansion in the first 200 turns).

## Troubleshooting
- **Oscillation:** If the manager flips modes too frequently, increase `macro_interval`.
- **Mode Neglect:** If a mode is never chosen, check the worker implementation for bugs or high failure rates.
