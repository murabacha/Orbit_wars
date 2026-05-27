"""
Behavioral Cloning trainer for pre-training the Relational Source-Target Transformer policy.
Synchronized to process higher-dimensional action tensor configurations cleanly.
"""
import argparse
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Enforce clean path insertions for local package lookups
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel


class BCDataset(Dataset):
    """ Ingests compressed structural trajectory data and validates feature layouts. """
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Target Behavioral Cloning dataset not found at: {npz_path}")
            
        data = np.load(npz_path)
        self.entities = data['entities']       # Shape: [N, Max_Entities, 18]
        self.entity_ids = data['entity_ids']   # Shape: [N, Max_Entities]
        self.mask = data['mask']               # Shape: [N, Max_Entities]
        
        # Load targets (aligned with full multi-discrete source-to-destination routing arrays)
        self.target = data['target']           # Shape: [N, Max_Entities]
        self.alloc = data['alloc']             # Shape: [N, Max_Entities]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            'entities': self.entities[idx],
            'entity_ids': self.entity_ids[idx],
            'mask': self.mask[idx],
            'target': self.target[idx],
            'alloc': self.alloc[idx]
        }


def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """ Batches variable lists into highly optimized parallel computational tensors. """
    return {
        'entities': torch.tensor(np.stack([b['entities'] for b in batch]), dtype=torch.float32),
        'entity_ids': torch.tensor(np.stack([b['entity_ids'] for b in batch]), dtype=torch.long),
        'mask': torch.tensor(np.stack([b['mask'] for b in batch]), dtype=torch.float32),
        'target': torch.tensor(np.stack([b['target'] for b in batch]), dtype=torch.long),
        'alloc': torch.tensor(np.stack([b['alloc'] for b in batch]), dtype=torch.long)
    }


def train_bc(npz_path: str, epochs: int = 5, batch_size: int = 64, lr: float = 3e-4, 
             device: str = 'cpu', save_path: str = 'checkpoints/bc_pretrained.pt'):
    
    print(f"Initializing Behavioral Cloning pre-training pipeline on device: {device}")
    dataset = BCDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # Initialize model using verified 18-dimensional predictive token definitions
    model = TransformerPPOModel(
        feature_dim=18, 
        embed_dim=128, 
        num_heads=4, 
        num_layers=3, 
        max_entities=200
    )
    model.to(device)

    # Instantiate specialized optimization parameters
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use reduction='none' to permit clean masked element filtering before pooling losses
    ce_loss = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        correct_targets = 0
        correct_allocs = 0
        total_valid_actions = 0

        for batch in dataloader:
            entities = batch['entities'].to(device)
            entity_ids = batch['entity_ids'].to(device)
            mask = batch['mask'].to(device)             # Shape: [B, N]
            targets = batch['target'].to(device)         # Shape: [B, N] (Target index per source node)
            allocs = batch['alloc'].to(device)           # Shape: [B, N] (Allocation index per source node)

            B, N, _ = entities.shape

            # Compute relational forward passes across grid paths
            target_logits, alloc_logits, _ = model(entities, entity_ids, mask)
            # Shapes: target_logits -> [B, N, N], alloc_logits -> [B, N, N, 6]

            # Construct logical operational masks to isolate real source assets owned by the agent
            # Observation processor maps friendly status to index vector channel 2 (owner == 0)
            is_source_owned = (entities[:, :, 2] == 1.0) # Evaluates boolean flags shape [B, N]
            valid_source_mask = is_source_owned & (mask == 1.0) # Avoid padding node evaluation
            
            if not valid_source_mask.any():
                continue # Safeguard pipeline execution against blank frame instances

            # 1. Calculate Relational Targeting Loss components
            # Flatten relational tensors into standard Cross-Entropy format
            # target_logits input format required: [B * N, N], targets required: [B * N]
            flat_target_logits = target_logits.view(-1, N)
            flat_targets = targets.view(-1)
            
            raw_target_loss = ce_loss(flat_target_logits, flat_targets) # Shape: [B * N]
            masked_target_loss = raw_target_loss * valid_source_mask.view(-1).float()
            
            # 2. Calculate Relational Allocation Loss components
            # We select the allocation logit slice that corresponds to the TRUE target selection index
            # This isolates errors along the matching trajectory choice path
            batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
            source_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
            chosen_target_indices = targets.view(-1)
            
            # Extract precise allocation slices shape: [B * N, 6]
            selected_alloc_logits = alloc_logits[batch_indices, source_indices, chosen_target_indices, :]
            flat_allocs = allocs.view(-1)
            
            raw_alloc_loss = ce_loss(selected_alloc_logits, flat_allocs) # Shape: [B * N]
            masked_alloc_loss = raw_alloc_loss * valid_source_mask.view(-1).float()

            # Normalize across valid entries to evaluate loss derivatives accurately
            n_valid = valid_source_mask.sum()
            loss = (masked_target_loss.sum() + masked_alloc_loss.sum()) / n_valid

            # Execute standard backpropagation cycles
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Compile execution logs metrics for precision tracking
            with torch.no_grad():
                # Extract argmax predictions along valid nodes
                pred_targets = flat_target_logits.argmax(dim=-1)
                pred_allocs = selected_alloc_logits.argmax(dim=-1)
                
                correct_targets += ((pred_targets == flat_targets) * valid_source_mask.view(-1)).sum().item()
                correct_allocs += ((pred_allocs == flat_allocs) * valid_source_mask.view(-1)).sum().item()
                total_valid_actions += n_valid.item()

            total_loss += loss.item() * B
            total_samples += B

        avg_loss = total_loss / max(1, total_samples)
        target_acc = (correct_targets / max(1, total_valid_actions)) * 100.0
        alloc_acc = (correct_allocs / max(1, total_valid_actions)) * 100.0
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} -> Loss: {avg_loss:.6f} | Target Acc: {target_acc:.2f}% | Alloc Acc: {alloc_acc:.2f}%")

    # Export finalized pre-trained checkpoint weight dict structures
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Successfully finalized and exported BC pretrained weights to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='checkpoints/bc_pretrained.pt')
    args = parser.parse_args()

    train_bc(
        npz_path=args.npz_path, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        device=args.device, 
        save_path=args.save_path
    )
