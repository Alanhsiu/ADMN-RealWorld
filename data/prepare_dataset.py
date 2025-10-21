"""
Split RGB-D dataset into train/val/test sets
Each sample consists of: color_image_XXXX.png + depth_image_XXXX.png
"""

import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import random


def get_sample_ids(class_dir):
    """
    Extract sample IDs from color image filenames
    E.g., 'color_image_0001.png' -> '0001'
    """
    files = os.listdir(class_dir)
    color_files = [f for f in files if f.startswith('color_image_')]
    
    # Extract IDs (e.g., '0001' from 'color_image_0001.png')
    sample_ids = [f.replace('color_image_', '').replace('.png', '') for f in color_files]
    sample_ids.sort()
    
    return sample_ids


def copy_sample_pair(sample_id, src_dir, dst_dir):
    """
    Copy both RGB and Depth images for a given sample ID
    """
    color_src = os.path.join(src_dir, f'color_image_{sample_id}.png')
    depth_src = os.path.join(src_dir, f'depth_image_{sample_id}.png')
    
    color_dst = os.path.join(dst_dir, f'color_image_{sample_id}.png')
    depth_dst = os.path.join(dst_dir, f'depth_image_{sample_id}.png')
    
    if os.path.exists(color_src) and os.path.exists(depth_src):
        shutil.copy2(color_src, color_dst)
        shutil.copy2(depth_src, depth_dst)
    else:
        print(f"⚠️  Warning: Missing files for sample {sample_id}")


def split_dataset(raw_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        raw_dir: Path to data/simple_raw_data/
        output_dir: Path to data/processed/
        train_ratio: Training set ratio (default 0.7 = 70%)
        val_ratio: Validation set ratio (default 0.15 = 15%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
    splits = ['train', 'val', 'test']
    
    # Create output directories
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    print("\n" + "="*60)
    print("📊 Dataset Split Summary")
    print("="*60)
    
    stats = {}
    
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        
        # Get all sample IDs
        sample_ids = get_sample_ids(cls_dir)
        n_samples = len(sample_ids)
        
        if n_samples == 0:
            print(f"⚠️  Warning: No samples found in {cls_dir}")
            continue
        
        # Calculate split sizes
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        # Split into train/val/test
        train_ids, temp_ids = train_test_split(
            sample_ids, 
            train_size=n_train,
            random_state=seed
        )
        
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=n_val,
            random_state=seed
        )
        
        # Copy files to respective directories
        for sample_id in train_ids:
            copy_sample_pair(sample_id, cls_dir, os.path.join(output_dir, 'train', cls))
        
        for sample_id in val_ids:
            copy_sample_pair(sample_id, cls_dir, os.path.join(output_dir, 'val', cls))
        
        for sample_id in test_ids:
            copy_sample_pair(sample_id, cls_dir, os.path.join(output_dir, 'test', cls))
        
        stats[cls] = {
            'total': n_samples,
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids)
        }
        
        print(f"{cls:15s} | Total: {n_samples:2d} | Train: {len(train_ids):2d} | Val: {len(val_ids):2d} | Test: {len(test_ids):2d}")
    
    # Save split info
    with open(os.path.join(output_dir, 'split_info.txt'), 'w') as f:
        f.write(f"Random seed: {seed}\n")
        f.write(f"Split ratio - Train: {train_ratio}, Val: {val_ratio}, Test: {1-train_ratio-val_ratio}\n\n")
        for cls, info in stats.items():
            f.write(f"{cls}: {info}\n")
    
    print("="*60)
    print(f"✅ Dataset split completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Split info saved to: {output_dir}/split_info.txt")
    print("="*60 + "\n")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split RGB-D dataset into train/val/test')
    parser.add_argument('--raw_dir', type=str, default='data/simple_raw_data',
                        help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("Train ratio + Val ratio cannot exceed 1.0")
    
    split_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )