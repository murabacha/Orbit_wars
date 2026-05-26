# Ablation Plan

To understand which components contribute most to the agent's performance, we will perform the following ablations:

1. **Remove Self-Attention:** Replace Transformer with a simple MLP to test spatial reasoning necessity.
2. **Remove Intercept Solver:** Force the RL agent to output continuous angles (0-2π).
3. **Remove ID Embedding:** Test if the agent can still learn list-order bias ("Geometric Tunneling") without explicit ID features.
4. **Remove Action Masking:** Allow the agent to target invalid objects and see how it affects training stability.
