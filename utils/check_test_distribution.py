"""
Check corruption distribution in train/val/test splits
"""

import torch
from scripts.train_stage2 import CorruptedGestureDataset, get_corrupted_dataloaders
from data.gesture_dataset import rgb_transform, depth_transform

# Load dataset
dataset = CorruptedGestureDataset('data', rgb_transform, depth_transform)

# Split (same as training)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Count corruption types in each split
def count_corruption(split_dataset, split_name):
    corruption_counts = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    
    for idx in split_dataset.indices:
        sample = dataset.samples[idx]
        corruption_counts[sample['corruption_type']] += 1
    
    print(f"\n{split_name} Set ({len(split_dataset)} samples):")
    for corr_type, count in corruption_counts.items():
        print(f"  {corr_type:15s}: {count:3d} samples ({100*count/len(split_dataset):.1f}%)")

print("="*60)
print("CORRUPTION DISTRIBUTION IN SPLITS")
print("="*60)

count_corruption(train_dataset, "Train")
count_corruption(val_dataset, "Val")
count_corruption(test_dataset, "Test")

print("\n" + "="*60)