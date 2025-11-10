"""
RGB-D Gesture Dataset (refer to GTDM PickleDataset.py)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ClampImg:
    """Clamp image values between 0 and 1"""
    def __call__(self, input):
        return torch.clamp(input, min=0, max=1)


class ExpandChannels:
    """Expand 1-channel depth to 3-channel for ViT"""
    def __call__(self, input):
        if len(input.shape) == 2:  # [H, W]
            input = torch.unsqueeze(input, dim=0)  # [1, H, W]
        if input.shape[0] == 1:  # [1, H, W]
            input = input.repeat(3, 1, 1)  # [3, H, W]
        return input


# Transforms for RGB and Depth (refer to the transform_dict in GTDM PickleDataset.py)
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ClampImg(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

depth_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ClampImg(),
    ExpandChannels(),  # Expand to 3 channels
])


class GestureRGBDDataset(Dataset):
    """
    RGB-D Dataset for gesture recognition
    Similar structure to PickleDataset but loads from image files
    """
    
    def __init__(self, root_dir, rgb_transform=None, depth_transform=None):
        """
        Args:
            root_dir: Path to data directory (e.g., 'data/processed/train')
            rgb_transform: Transformations for RGB images
            depth_transform: Transformations for depth images
        """
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform or rgb_transform
        self.depth_transform = depth_transform or depth_transform
        
        # Class mapping
        self.classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all samples into memory (like PickleDataset)
        print(f"Loading dataset from {root_dir}...")
        self.data = self._load_all_samples()
        print(f"Loaded {len(self.data)} samples")
    
    def _load_all_samples(self):
        """Load all RGB-D pairs into memory"""
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found")
                continue
            
            # Get all color images
            color_files = sorted([f for f in os.listdir(class_dir) 
                                 if f.startswith('color_image_')])
            
            for color_file in color_files:
                sample_id = color_file.replace('color_image_', '').replace('.png', '')
                
                color_path = os.path.join(class_dir, color_file)
                depth_path = os.path.join(class_dir, f'depth_image_{sample_id}.png')
                
                if os.path.exists(color_path) and os.path.exists(depth_path):
                    # Load images into memory
                    rgb_img = Image.open(color_path).convert('RGB')
                    depth_img = Image.open(depth_path).convert('L')
                    
                    samples.append({
                        'rgb': rgb_img,
                        'depth': depth_img,
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name,
                        'sample_id': sample_id
                    })
        
        return samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            data: dict with 'rgb', 'depth' tensors
            label: class label (int)
        """
        sample = self.data[idx]
        
        # Apply transforms
        rgb = self.rgb_transform(sample['rgb'])
        depth = self.depth_transform(sample['depth'])
        
        # Return format similar to GTDM (data dict + label)
        data = {
            'rgb': rgb,
            'depth': depth,
        }
        
        return data, sample['label']

def single_image_transform(rgb_image_path, depth_image_path):
    """Transform a single RGB-D image pair for inference"""
    rgb_img = Image.open(rgb_image_path).convert('RGB')
    depth_img = Image.open(depth_image_path).convert('L')
    
    rgb = rgb_transform(rgb_img)
    depth = depth_transform(depth_img)
    
    data = {
        'rgb': rgb.unsqueeze(0),  # Add batch dimension
        'depth': depth.unsqueeze(0),
    }
    
    return data

def get_dataloaders(data_dir='data/processed', batch_size=32, num_workers=4):
    """
    Create train/val/test dataloaders
    Similar to the dataloader creation in train.py
    """
    train_dataset = GestureRGBDDataset(
        root_dir=os.path.join(data_dir, 'train'),
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    val_dataset = GestureRGBDDataset(
        root_dir=os.path.join(data_dir, 'val'),
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    test_dataset = GestureRGBDDataset(
        root_dir=os.path.join(data_dir, 'test'),
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


# Test the dataset
if __name__ == "__main__":
    print("Testing GestureRGBDDataset...")
    
    # Create dataset
    train_dataset = GestureRGBDDataset(
        root_dir='data/processed/train',
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    print(f"\nDataset size: {len(train_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Test one sample
    data, label = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  RGB shape: {data['rgb'].shape}")
    print(f"  Depth shape: {data['depth'].shape}")
    print(f"  Label: {label} ({train_dataset.classes[label]})")
    
    # Test dataloader
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    
    data_batch, labels_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  RGB: {data_batch['rgb'].shape}")
    print(f"  Depth: {data_batch['depth'].shape}")
    print(f"  Labels: {labels_batch.shape}")
    
    print("\n✅ Dataset test passed!")