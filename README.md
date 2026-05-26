# Orbit Wars AI: Competitive Agent Development Kit

This repository contains a complete reinforcement learning pipeline for building high-performance agents for the Orbit Wars environment.

## Project Structure
- `agents/`: Implementation of Candidate architectures (Transformer-PPO, Hierarchical).
- `environment/`: Critical wrappers for intercept math, observation tokenization, and reward shaping.
- `training/`: Scripts for PPO training, Self-Play management, and Curriculum learning.
- `docs/`: Architectural design documents and training specifications.
- `tests/`: Unit tests for core environment math.

## Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training
To begin training the Transformer-PPO agent:
```bash
python training/train_ppo.py
```

### 3. Evaluation
To run a local tournament against baseline heuristics:
```bash
python training/evaluation.py
```

## Core Technologies
- **PPO:** Stabilized policy updates.
- **Transformer Encoder:** Global spatial attention.
- **Intercept Solver:** 8-iteration mathematical precision for moving planets.
- **League Training:** Robustness against non-stationary opponents.

## Research Foundation
All design decisions are based on the [Research Summary](docs/research_summary.md).
