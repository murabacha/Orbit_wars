import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPPOModel(nn.Module):
    """
    Optimized Relational Transformer for Orbit Wars AI.
    Memory-efficient architecture designed for 200+ entities on standard GPUs.
    Supports FP16/AMP without logit overflow.
    """
    def __init__(self, feature_dim: int = 18, embed_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 3, max_entities: int = 200, max_simulation_id: int = 2000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_entities = max_entities
        
        self.entity_embedding = nn.Linear(feature_dim, embed_dim)
        self.id_embedding = nn.Embedding(max_simulation_id, embed_dim, padding_idx=0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=256, 
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.target_src_proj = nn.Linear(embed_dim, 64)
        self.target_tgt_proj = nn.Linear(embed_dim, 64)
        
        self.target_score_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        self.allocation_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 101)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, entities: torch.Tensor, entity_ids: torch.Tensor, 
                mask: torch.Tensor, action_masks: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, F_dim = entities.shape
        
        clamped_ids = torch.clamp(entity_ids, 0, self.id_embedding.num_embeddings - 1)
        x = self.entity_embedding(entities) + self.id_embedding(clamped_ids)
        
        padding_mask = (mask == 0)
        latent = self.transformer(x, src_key_padding_mask=padding_mask) 
        
        mask_expanded = mask.unsqueeze(-1)
        masked_latent = latent * mask_expanded
        global_latent = masked_latent.sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0)
        
        src_features = self.target_src_proj(latent).unsqueeze(2)
        tgt_features = self.target_tgt_proj(latent).unsqueeze(1)
        
        relational_grid = F.gelu(src_features + tgt_features)
        
        target_logits = self.target_score_head(relational_grid).squeeze(-1)
        allocation_logits = self.allocation_head(relational_grid)
        
        if action_masks is not None:
            # FIX: Use -1e4 instead of -1e9 to avoid FP16 (Half) overflow in AMP mode
            target_logits = target_logits.masked_fill(~action_masks, -1e4)
            
        # FIX 5: Detach global_latent to protect shared trunk from value loss explosions
        value = self.critic_head(global_latent.detach())
        return target_logits, allocation_logits, value

def load_checkpoint_with_surgery(model: nn.Module, path: str, device: torch.device) -> None:
    state_dict = torch.load(path, map_location=device)
    model_state = model.state_dict()
    
    if 'allocation_head.2.weight' in state_dict:
        ckpt_weight = state_dict['allocation_head.2.weight']
        ckpt_bias = state_dict['allocation_head.2.bias']
        model_weight = model_state['allocation_head.2.weight']
        
        if ckpt_weight.shape != model_weight.shape:
            # We have a shape mismatch. We perform weight surgery!
            # ckpt_weight shape is (6, 64), model_weight shape is (101, 64)
            new_weight = torch.zeros_like(model_weight)
            new_bias = torch.zeros_like(model_state['allocation_head.2.bias'])
            
            # Map discrete percentage indices to hybrid 101 bins
            # Bins 0-75 represent absolute counts 0 to 75
            # Bins 76-100 represent percentages 0% to 100% (mapped to old indices 0 to 4)
            
            # 1. Bins 0-75 (absolute counts): interpolate from 0% (old idx 0) to 25% (old idx 1)
            for i in range(76):
                frac = i / 75.0
                new_weight[i] = (1.0 - frac) * ckpt_weight[0] + frac * ckpt_weight[1]
                new_bias[i] = (1.0 - frac) * ckpt_bias[0] + frac * ckpt_bias[1]
                
            # 2. Bins 76-82: interpolate from old idx 0 (0%) to old idx 1 (25%)
            for i in range(76, 83):
                frac = (i - 76) / 6.0
                new_weight[i] = (1.0 - frac) * ckpt_weight[0] + frac * ckpt_weight[1]
                new_bias[i] = (1.0 - frac) * ckpt_bias[0] + frac * ckpt_bias[1]
                
            # 3. Bins 82-88: interpolate from old idx 1 (25%) to old idx 2 (50%)
            for i in range(82, 89):
                frac = (i - 82) / 6.0
                new_weight[i] = (1.0 - frac) * ckpt_weight[1] + frac * ckpt_weight[2]
                new_bias[i] = (1.0 - frac) * ckpt_bias[1] + frac * ckpt_bias[2]
                
            # 4. Bins 88-94: interpolate from old idx 2 (50%) to old idx 3 (75%)
            for i in range(88, 95):
                frac = (i - 88) / 6.0
                new_weight[i] = (1.0 - frac) * ckpt_weight[2] + frac * ckpt_weight[3]
                new_bias[i] = (1.0 - frac) * ckpt_bias[2] + frac * ckpt_bias[3]
                
            # 5. Bins 94-100: interpolate from old idx 3 (75%) to old idx 4 (100%)
            for i in range(94, 101):
                frac = (i - 94) / 6.0
                new_weight[i] = (1.0 - frac) * ckpt_weight[3] + frac * ckpt_weight[4]
                new_bias[i] = (1.0 - frac) * ckpt_bias[3] + frac * ckpt_bias[4]
                
            state_dict['allocation_head.2.weight'] = new_weight
            state_dict['allocation_head.2.bias'] = new_bias
            
    model.load_state_dict(state_dict)
