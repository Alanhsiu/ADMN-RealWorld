"""
Verify dataset integrity after splitting
"""

import os
import argparse
from pathlib import Path


def verify_dataset(data_dir):
    """
    Check if dataset is properly structured and complete
    """
    classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
    splits = ['train', 'val', 'test']
    
    print("\n" + "="*60)
    print("🔍 Dataset Verification")
    print("="*60)
    
    all_good = True
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        for cls in classes:
            cls_dir = os.path.join(data_dir, split, cls)
            
            if not os.path.exists(cls_dir):
                print(f"  ❌ {cls}: Directory not found!")
                all_good = False
                continue
            
            # Count color and depth images
            files = os.listdir(cls_dir)
            color_files = [f for f in files if f.startswith('color_image_')]
            depth_files = [f for f in files if f.startswith('depth_image_')]
            
            n_color = len(color_files)
            n_depth = len(depth_files)
            
            # Check if RGB-Depth pairs match
            if n_color == n_depth:
                print(f"  ✅ {cls:15s}: {n_color} RGB-D pairs")
            else:
                print(f"  ⚠️  {cls:15s}: {n_color} RGB, {n_depth} Depth (MISMATCH!)")
                all_good = False
            
            # Verify each pair exists
            color_ids = set([f.replace('color_image_', '').replace('.png', '') for f in color_files])
            depth_ids = set([f.replace('depth_image_', '').replace('.png', '') for f in depth_files])
            
            missing_depth = color_ids - depth_ids
            missing_color = depth_ids - color_ids
            
            if missing_depth:
                print(f"     ⚠️  Missing depth for IDs: {missing_depth}")
                all_good = False
            if missing_color:
                print(f"     ⚠️  Missing color for IDs: {missing_color}")
                all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✅ All checks passed! Dataset is ready for training.")
    else:
        print("❌ Some issues found. Please check the warnings above.")
    print("="*60 + "\n")
    
    return all_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    
    args = parser.parse_args()
    
    verify_dataset(args.data_dir)