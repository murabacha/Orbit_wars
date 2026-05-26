"""
Behavioral Cloning trainer for pre-training the Transformer policy on heuristic rollouts.

Loads the dataset produced by `generate_bc_data.py` and optimizes the policy's
`target_head` and `allocation_head` using cross-entropy losses.

Saves pretrained weights to `checkpoints/bc_pretrained.pt`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from orbit_wars_ai.agents.transformer_ppo.model import TransformerPPOModel
from orbit_wars_ai.agents.transformer_ppo.config import TransformerPPOConfig


class BCDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.entities = data['entities']
        self.entity_ids = data['entity_ids']
        self.mask = data['mask']
        self.target = data['target']
        self.alloc = data['alloc']

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            'entities': self.entities[idx],
            'entity_ids': self.entity_ids[idx],
            'mask': self.mask[idx],
            'target': int(self.target[idx]),
            'alloc': int(self.alloc[idx])
        }


def collate_fn(batch):
    entities = np.stack([b['entities'] for b in batch])
    entity_ids = np.stack([b['entity_ids'] for b in batch])
    mask = np.stack([b['mask'] for b in batch])
    targets = np.array([b['target'] for b in batch])
    allocs = np.array([b['alloc'] for b in batch])
    return {
        'entities': torch.tensor(entities, dtype=torch.float32),
        'entity_ids': torch.tensor(entity_ids, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.float32),
        'target': torch.tensor(targets, dtype=torch.long),
        'alloc': torch.tensor(allocs, dtype=torch.long)
    }


def train_bc(npz_path, epochs=5, batch_size=64, lr=3e-4, device='cpu', save_path='checkpoints/bc_pretrained.pt'):
    config = TransformerPPOConfig()
    dataset = BCDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = TransformerPPOModel(feature_dim=config.feature_dim, embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers, max_entities=config.max_entities)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            entities = batch['entities'].to(device)
            entity_ids = batch['entity_ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target'].to(device)
            allocs = batch['alloc'].to(device)

            target_logits, alloc_logits, _ = model(entities, entity_ids, mask)
            # target_logits: [batch, seq_len]
            # CrossEntropy expects [batch, C] where C=seq_len -> need to pass full logits and targets
            loss_target = ce_loss(target_logits, targets)
            loss_alloc = ce_loss(alloc_logits, allocs)
            loss = loss_target + loss_alloc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item() * entities.size(0)
            total_samples += entities.size(0)

        avg_loss = total_loss / max(1, total_samples)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved BC pretrained weights to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, default='data/bc_dataset/bc_data.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_path', type=str, default='checkpoints/bc_pretrained.pt')
    args = parser.parse_args()

    train_bc(args.npz_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, save_path=args.save_path)
