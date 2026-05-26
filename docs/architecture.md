# System Architecture

The Orbit Wars AI system is built on a modular design that separates strategic intent from physical execution.

## 1. Data Flow
1. **Raw Observation:** Received from Kaggle.
2. **ObservationProcessor:** Normalizes and tokenizes entities.
3. **Policy:** Transformer encoder processes tokens and outputs logits for Target and Allocation.
4. **ActionProcessor:** Translates discrete indices into continuous angles/ships.
5. **Wrapper:** Refines angles via 8-iteration intercept solver.

## 2. Transformer Backbone
The backbone uses a Standard Transformer Encoder. 
- **Embeddings:** Combine spatial (X,Y) and categorical (Owner, Type) features.
- **Attention:** Multi-head self-attention allows any planet to "see" any other fleet's threat level globally.
- **Output:** Decoupled heads for hierarchical action selection.

## 3. The Intercept Bridge
The most critical non-learned component is the **Intercept Solver**. It removes the need for the RL agent to learn trigonometry, focusing the gradient entirely on **strategic selection**.
