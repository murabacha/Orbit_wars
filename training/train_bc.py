"""
Behavioral Cloning trainer (Dual-Head Pre-training).
Trains both the Policy (Action Matching) and the Critic (Value Estimation) simultaneously.
"""
import argparse
import math
import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel

class BCDataset(Dataset):
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Target dataset not found: {npz_path}")
        data = np.load(npz_path)
        self.entities = data['entities']
        self.entity_ids = data['entity_ids']
        self.mask = data['mask']
        self.target = data['target']
        self.alloc = data['alloc']
        
        # Load returns, default to 0.0 if loading an old dataset
        try:
            self.returns = data['returns']
        except KeyError:
            print("⚠️ 'returns' not found in dataset. Critic will not train optimally.")
            self.returns = np.zeros(len(self.target), dtype=np.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx: int):
        return {
            'entities': self.entities[idx],
            'entity_ids': self.entity_ids[idx],
            'mask': self.mask[idx],
            'target': self.target[idx],
            'alloc': self.alloc[idx],
            'returns': self.returns[idx]
        }

def collate_fn(batch):
    return {
        'entities': torch.tensor(np.stack([b['entities'] for b in batch]), dtype=torch.float32),
        'entity_ids': torch.tensor(np.stack([b['entity_ids'] for b in batch]), dtype=torch.long),
        'mask': torch.tensor(np.stack([b['mask'] for b in batch]), dtype=torch.float32),
        'target': torch.tensor(np.stack([b['target'] for b in batch]), dtype=torch.long),
        'alloc': torch.tensor(np.stack([b['alloc'] for b in batch]), dtype=torch.long),
        'returns': torch.tensor(np.stack([b['returns'] for b in batch]), dtype=torch.float32)
    }

def train_bc(npz_path: str, epochs: int = 5, batch_size: int = 16, lr: float = 3e-4, 
             device: str = 'cpu', save_path: str = 'checkpoints/bc_pretrained.pt'):
    
    print(f"🚀 Initializing Dual-Head BC Trainer [Batch Size: {batch_size}, Device: {device}, LR: {lr}]")
    if 'cuda' in device:
        torch.cuda.empty_cache()
        gc.collect()

    dataset = BCDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    model = TransformerPPOModel(feature_dim=18, embed_dim=128, num_heads=4, num_layers=3, max_entities=200)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if 'cuda' in device else None
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    # NEW: Mitigate Dataset Imbalance (The "Hold Position" flood)
    # The heuristic almost never launches (alloc=0 is 95% of the data).
    alloc_weights = torch.ones(101, device=device)
    alloc_weights[0] = 0.05
    alloc_ce_loss = nn.CrossEntropyLoss(weight=alloc_weights, reduction='none')

    for epoch in range(epochs):
        model.train()
        total_loss, total_v_loss, total_valid = 0.0, 0.0, 0
        correct_targets, correct_allocs = 0, 0

        for batch in dataloader:
            entities = batch['entities'].to(device)
            entity_ids = batch['entity_ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].to(device)
            allocs = batch['alloc'].to(device)
            returns = batch['returns'].to(device) # Shape: [B]
            
            B, N = entities.shape[0], entities.shape[1]

            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # We now unpack all three: Target, Alloc, AND Values
                target_logits, alloc_logits, values = model(entities, entity_ids, mask)

                # valid_mask finds "My Planets" (Index 2 is "ME")
                is_source_owned = (entities[:, :, 2] == 1.0)
                valid_mask = is_source_owned & (mask == 1.0)
                
                if not valid_mask.any(): continue

                # --- TARGET FIXATION CURE ---
                # Find out if the expert actually pressed the launch button
                is_attacking = (allocs.view(-1) > 0).float()

                # 1. Targeting Loss (Actor)
                flat_target_logits = target_logits.view(-1, N)
                flat_targets = targets.view(-1)
                
                # Multiply by is_attacking so 'Hold' actions don't force Target=0
                t_loss = (ce_loss(flat_target_logits, flat_targets) * valid_mask.view(-1) * is_attacking).sum()
                
                # 2. Allocation Loss (Actor) - USING WEIGHTED LOSS
                batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
                source_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
                selected_alloc = alloc_logits[batch_idx, source_idx, targets.view(-1), :]
                a_loss = (alloc_ce_loss(selected_alloc, allocs.view(-1)) * valid_mask.view(-1)).sum()

                # 3. Value Estimation Loss (Critic) - Standard MSE
                v_loss = F.mse_loss(values.squeeze(-1), returns)

                # Decouple averages so the rare attack frames generate strong gradients
                n_count = valid_mask.sum()
                n_attacking = (valid_mask.view(-1) * is_attacking).sum()
                
                t_loss_mean = t_loss / torch.clamp(n_attacking, min=1.0)
                a_loss_mean = a_loss / torch.clamp(n_count, min=1.0)

                actor_loss = t_loss_mean + a_loss_mean
                loss = actor_loss + (0.5 * v_loss)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            with torch.no_grad():
                correct_targets += ((flat_target_logits.argmax(-1) == flat_targets) * valid_mask.view(-1)).sum().item()
                correct_allocs += ((selected_alloc.argmax(-1) == allocs.view(-1)) * valid_mask.view(-1)).sum().item()
                total_valid += n_count.item()
                total_loss += loss.item() * B
                total_v_loss += v_loss.item() * B

        if total_valid > 0:
            print(f"Epoch {epoch+1:02d} | Total Loss: {total_loss/len(dataset):.4f} | V_Loss (Critic): {total_v_loss/len(dataset):.4f} | Target Acc: {100*correct_targets/total_valid:.2f}% | Alloc Acc: {100*correct_allocs/total_valid:.2f}%")
        
        if 'cuda' in device:
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Dual-Head Pre-training complete. Saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
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
