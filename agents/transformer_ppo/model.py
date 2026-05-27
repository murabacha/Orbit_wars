import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPPOModel(nn.Module):
    """
    State-of-the-art Permutation-Invariant Transformer Architecture for Orbit Wars AI.
    Processes 18-dimensional predictive entity tokens and computes structured, source-decoupled
    multi-discrete target and ship allocation logits while strictly enforcing invalid action masking.
    """
    def __init__(self, feature_dim: int = 18, embed_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 3, max_entities: int = 200, max_simulation_id: int = 2000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_entities = max_entities
        
        # Continuous Feature Token Projection Head
        self.entity_embedding = nn.Linear(feature_dim, embed_dim)
        
        # High-Capacity Vocabulary Identity Embedding to map true Simulation IDs (Geometric Tunneling)
        self.id_embedding = nn.Embedding(max_simulation_id, embed_dim, padding_idx=0)
        
        # Standard Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoupled Source-Target Multi-Discrete Actor Heads
        # Resolves multi-front orchestration challenges by processing paired source-destination latents
        self.target_src_proj = nn.Linear(embed_dim, embed_dim)
        self.target_tgt_proj = nn.Linear(embed_dim, embed_dim)
        self.target_score_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Discretized Ship Allocation Head: [0=0%, 1=25%, 2=50%, 3=75%, 4=100%, 5=Exact Needed]
        self.allocation_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 6)
        )
        
        # Critic Head (Predicts state value using true masked mean pool states)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, entities: torch.Tensor, entity_ids: torch.Tensor, 
                mask: torch.Tensor, action_masks: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            entities: [batch, max_entities, feature_dim] (18 Features)
            entity_ids: [batch, max_entities] (True internal environment lookup IDs)
            mask: [batch, max_entities] (1.0 for valid real elements, 0.0 for padding rows)
            action_masks: [batch, max_entities, max_entities] Optional target legality tracking matrix
        Returns:
            target_logits: [batch, max_entities, max_entities] (Source-to-Target mapping decisions)
            allocation_logits: [batch, max_entities, max_entities, 6] (Allocation heads per valid pathway)
            value: [batch, 1] (Global Critic state value baseline estimation)
        """
        B, N, F_dim = entities.shape
        
        # 1. High-Fidelity Embedding Accumulation
        # Clamp true simulation tracking IDs to protect vocabulary limits from corrupt outliers
        clamped_ids = torch.clamp(entity_ids, 0, self.id_embedding.num_embeddings - 1)
        x = self.entity_embedding(entities) + self.id_embedding(clamped_ids)
        
        # 2. Sequence Transformer Block Processing
        # Invert mask matrix parameters to comply with PyTorch standard padding formats (True = Mask out)
        padding_mask = (mask == 0)
        latent = self.transformer(x, src_key_padding_mask=padding_mask) # Output shape: [B, N, embed_dim]
        
        # 3. Mathematical Masked Mean Pool Optimization (Resolves Invariance Flaws)
        # Isolate true valid tokens from sequence noise before computing value operations
        mask_expanded = mask.unsqueeze(-1) # [B, N, 1]
        masked_latent = latent * mask_expanded
        global_latent = masked_latent.sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1.0) # [B, embed_dim]
        
        # 4. Source-Target Structural Decoupling Execution
        # Broadcast all possible source and target node pairs to allow precise tactical routing
        src_features = self.target_src_proj(latent).unsqueeze(2)  # [B, N, 1, embed_dim]
        tgt_features = self.target_tgt_proj(latent).unsqueeze(1)  # [B, 1, N, embed_dim]
        
        # Combine relational features across shared coordinate boundaries
        relational_grid = F.gelu(src_features + tgt_features)     # [B, N, N, embed_dim]
        target_logits = self.target_score_head(relational_grid).squeeze(-1) # [B, N, N]
        
        # 5. Allocate Ship Mass Logistics per Source-Target Pair
        # Concatenate individual source properties to the relational grid to preserve ownership profiles
        src_broadcast = latent.unsqueeze(2).expand(-1, -1, N, -1) # [B, N, N, embed_dim]
        allocation_context = torch.cat([src_broadcast, relational_grid], dim=-1) # [B, N, N, embed_dim * 2]
        allocation_logits = self.allocation_head(allocation_context) # [B, N, N, 6]
        
        # 6. Apply Hard Structural Penalties via Logit Masking
        if action_masks is not None:
            # Set forbidden target pairs to negative infinity to prevent invalid action evaluation
            target_logits = target_logits.masked_fill(~action_masks, -1e9)
            
        # 7. Compute Global State Estimations via Critic Head
        value = self.critic_head(global_latent) # [B, 1]
        
        return target_logits, allocation_logits, value
