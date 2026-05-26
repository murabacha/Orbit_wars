# Deployment Guide

## 1. Packaging
For Kaggle, all dependencies must be pre-installed or included in the `main.py`.
- Weights should be encoded as a base64 string or included as a separate `.pt` file if the environment allows datasets.

## 2. Inference Constraints
- **Time:** 1.0s limit per act.
- **Memory:** 4GB RAM.
- **Batching:** Use `torch.no_grad()` and ensure the Transformer layer count doesn't exceed 4 to keep latency low.

## 3. Submission Format
```python
def agent(obs, config):
    # 1. Init processors
    # 2. Load weights (cached)
    # 3. Process obs -> Action
    return moves
```
