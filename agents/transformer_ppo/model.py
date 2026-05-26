import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPPOModel(nn.Module):
    """
    State-of-the-art Transformer architecture for Orbit Wars.
    Treats all entities as tokens in a self-attention sequence.
    """
    def __init__(self, feature_dim: int, embed_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 3, max_entities: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_entities = max_entities
        
        # Entity Embedding
        self.entity_embedding = nn.Linear(feature_dim, embed_dim)
        
        # Positional/ID Embedding (Crucial for learning Geometric Tunneling quirks)
        self.id_embedding = nn.Embedding(max_entities + 1, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Actor Heads (Multi-discrete outputs)
        # 1. Target Logits (One for each entity in the sequence)
        self.target_head = nn.Linear(embed_dim, 1)
        
        # 2. Allocation Logits [0%, 25%, 50%, 75%, 100%, exact_needed]
        self.allocation_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
        # Critic Head (Global state value)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, entities: torch.Tensor, entity_ids: torch.Tensor, mask: torch.Tensor):
        """
        entities: [batch, seq_len, feature_dim]
        entity_ids: [batch, seq_len]
        mask: [batch, seq_len] (1 for valid, 0 for padding)
        """
        # 1. Embedding
        x = self.entity_embedding(entities) + self.id_embedding(entity_ids)
        
        # 2. Transformer Attention
        # Create src_key_padding_mask (True for padding/masked)
        padding_mask = (mask == 0)
        latent = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # 3. Actor: Target Selection
        target_logits = self.target_head(latent).squeeze(-1) # [batch, seq_len]
        
        # 4. Actor: Allocation (Using global pool or per-entity)
        # We pool the latent to get a global context for allocation
        global_latent = latent.mean(dim=1)
        allocation_logits = self.allocation_head(global_latent) # [batch, 6]
        
        # 5. Critic: Value
        value = self.critic_head(global_latent)
        
        return target_logits, allocation_logits, value
