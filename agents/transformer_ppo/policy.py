import torch
import torch.distributions as dist
from .model import TransformerPPOModel
from .config import TransformerPPOConfig

class TransformerPPOPolicy:
    """
    Policy wrapper that handles sampling and log-probabilities for the Transformer model.
    """
    def __init__(self, config: TransformerPPOConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.model = TransformerPPOModel(
            feature_dim=config.feature_dim,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_entities=config.max_entities
        ).to(device)

    def get_action(self, obs_tensors, action_mask=None):
        """
        Samples an action from the policy.
        """
        entities = torch.tensor(obs_tensors['entities']).unsqueeze(0).to(self.device)
        entity_ids = torch.tensor(obs_tensors['entity_ids']).unsqueeze(0).to(self.device)
        mask = torch.tensor(obs_tensors['mask']).unsqueeze(0).to(self.device)

        B, N, _ = entities.shape

        with torch.no_grad():
            target_logits, allocation_logits, value = self.model(entities, entity_ids, mask, action_masks=action_mask)

            target_dist = dist.Categorical(logits=target_logits)
            target_action_flat = target_dist.sample() # [1]

            # Decode flattened index to (source, target)
            target_action_idx = target_action_flat.item()
            source_idx = target_action_idx // N
            target_idx = target_action_idx % N

            # Sample allocation strictly for the chosen path
            # target_logits is [1, N*N], allocation_logits is [1, N*N, 6]
            selected_alloc_logits = allocation_logits[0, target_action_idx, :]
            allocation_dist = dist.Categorical(logits=selected_alloc_logits)
            allocation_action = allocation_dist.sample()

            log_prob = target_dist.log_prob(target_action_flat) + allocation_dist.log_prob(allocation_action)

        return source_idx, target_idx, allocation_action.item(), log_prob.item(), value.item()


    def evaluate_actions(self, entities, entity_ids, mask, target_actions, allocation_actions):
        """
        Evaluates actions for PPO updates.
        """
        target_logits, allocation_logits, values = self.model(entities, entity_ids, mask)
        
        target_dist = dist.Categorical(logits=target_logits)
        allocation_dist = dist.Categorical(logits=allocation_logits)
        
        log_probs = target_dist.log_prob(target_actions) + allocation_dist.log_prob(allocation_actions)
        entropy = target_dist.entropy() + allocation_dist.entropy()
        
        return log_probs, values, entropy
