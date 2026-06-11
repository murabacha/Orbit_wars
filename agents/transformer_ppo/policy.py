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
        Samples a vector of actions (one per entity) from the policy.
        """
        entities = torch.tensor(obs_tensors['entities']).unsqueeze(0).to(self.device)
        entity_ids = torch.tensor(obs_tensors['entity_ids']).unsqueeze(0).to(self.device)
        mask = torch.tensor(obs_tensors['mask']).unsqueeze(0).to(self.device)

        B, N, _ = entities.shape

        if action_mask is not None:
            action_mask = torch.tensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            target_logits, allocation_logits, value = self.model(entities, entity_ids, mask, action_masks=action_mask)

            # target_logits: [1, N, N], allocation_logits: [1, N, N, 6]
            target_dist = dist.Categorical(logits=target_logits)
            target_actions = target_dist.sample() # [1, N]

            # Sample allocations for all chosen targets
            # We sample from the allocation distribution of EVERY source node
            batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, N).reshape(-1)
            source_idx = torch.arange(N, device=self.device).unsqueeze(0).expand(B, -1).reshape(-1)
            chosen_targets = target_actions.view(-1)

            selected_alloc_logits = allocation_logits[batch_idx, source_idx, chosen_targets, :]
            allocation_dist = dist.Categorical(logits=selected_alloc_logits)
            allocation_actions = allocation_dist.sample().view(B, N)

            # Calculate log_probs strictly for owned assets
            is_source_owned = (entities[:, :, 2] == 1.0)
            valid_source_mask = is_source_owned & (mask == 1.0)

            log_p_target = target_dist.log_prob(target_actions) # [1, N]
            log_p_alloc = allocation_dist.log_prob(allocation_actions.view(-1)).view(B, N)

            # FIX 1: Sum log probs to reflect joint action probability. Do NOT divide by valid_counts.
            joint_log_prob = ((log_p_target + log_p_alloc) * valid_source_mask.float()).sum(dim=-1)

        return target_actions.squeeze(0).cpu().numpy(), allocation_actions.squeeze(0).cpu().numpy(), joint_log_prob.item(), value.item()

    def evaluate_actions(self, entities, entity_ids, mask, action_masks, target_actions, allocation_actions):
        """
        Evaluates joint policy distributions for PPO updates.
        """
        target_logits, allocation_logits, values = self.model(entities, entity_ids, mask, action_masks=action_masks)

        B, N, _ = entities.shape

        target_dist = dist.Categorical(logits=target_logits)

        batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(-1, N).reshape(-1)
        source_idx = torch.arange(N, device=self.device).unsqueeze(0).expand(B, -1).reshape(-1)
        chosen_targets = target_actions.view(-1)

        selected_alloc_logits = allocation_logits[batch_idx, source_idx, chosen_targets, :]
        allocation_dist = dist.Categorical(logits=selected_alloc_logits)

        is_source_owned = (entities[:, :, 2] == 1.0)
        valid_source_mask = is_source_owned & (mask == 1.0)

        log_p_target = target_dist.log_prob(target_actions)
        log_p_alloc = allocation_dist.log_prob(allocation_actions.view(-1)).view(B, N)

        # FIX 1: Sum log probs to reflect joint action probability
        joint_log_prob = ((log_p_target + log_p_alloc) * valid_source_mask.float()).sum(dim=-1)
        
        # Keep the mean strictly for the entropy bonus calculation
        entropy = ((target_dist.entropy() + allocation_dist.entropy().view(B, N)) * valid_source_mask.float()).sum(dim=-1) / torch.clamp(valid_source_mask.sum(dim=-1), min=1.0)

        return joint_log_prob, values.squeeze(-1), entropy.mean()
