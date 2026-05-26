# Agent Comparison

| Feature | Transformer-PPO | Hierarchical RL | Heuristic Baseline |
| :--- | :--- | :--- | :--- |
| **Logic** | Neural (Learned) | Hybrid (Manager/Worker) | Rule-based |
| **Input** | Raw Tokenized Entities | Aggregated Stats | Structured Objects |
| **Horizon** | 500 Steps (GAE) | 10 Step Macro Blocks | Infinite/Greedy |
| **Inference Speed** | Slow (~100ms) | Moderate (~50ms) | Fast (<5ms) |
| **Flexibility** | High (Emergent tactics) | Medium (Constrained) | Low (Static) |
| **Training Cost** | Very High | High | Zero |

### Recommendation
**Transformer-PPO** is the primary choice for the tournament. Its ability to process permutation-invariant entity sets and use self-attention allows it to discover timing-based exploits (like swarming) that rules cannot easily capture.
