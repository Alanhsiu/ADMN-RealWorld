"""
Common dataloader creation for both Stage 1 and Stage 2
Ensures consistent train/val split
"""

import torch
from torch.utils.data import DataLoader
from scripts.train_stage2 import CorruptedGestureDataset
from data.gesture_dataset import rgb_transform, depth_transform


def get_consistent_dataloaders(data_dir, batch_size=16, num_workers=4, seed=42):
    """
    Create train/val dataloaders with consistent split for both stages
    
    Returns:
        train_loader, val_loader, test_loader (test = val)
    """
    # Create full dataset
    dataset = CorruptedGestureDataset(
        root_dir=data_dir,
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    print(f"\nDataset: {len(dataset)} samples")
    
    # 80/20 split with fixed seed
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Split (80/20):")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Return val as test (no separate test set)
    return train_loader, val_loader, val_loader