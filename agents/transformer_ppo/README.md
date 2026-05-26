# Transformer PPO Agent

This agent uses a Transformer architecture to process the variable-length set of entities (planets, fleets, comets) in the Orbit Wars environment.

## Architecture
- **State Processor:** `ObservationProcessor` converts the game state into a tokenized sequence.
- **Backbone:** 3-layer Transformer Encoder with 4 attention heads.
- **Actor:** Multi-discrete heads for `Target Entity ID` and `Ship Allocation`.
- **Critic:** MLP head predicting state value from the pooled Transformer latent.

## Key Features
- **Permutation Invariance:** The agent can process any number of planets and fleets in any order.
- **Global Context:** Self-attention allows the agent to recognize cross-map threats instantly.
- **Discrete Intent:** Decouples high-level strategic choice from low-level physics (handled by the wrapper).

## Usage
Train this agent using the `training/train_ppo.py` script.
Initial training should use Behavioral Cloning to establish basic physics competency.
