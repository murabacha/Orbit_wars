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
            nn.Linear(64, 6)
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
