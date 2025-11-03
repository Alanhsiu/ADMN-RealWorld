"""
Stage 2 Training: Train Adaptive Controller
Learn to allocate layers based on RGB-D quality
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import rgb_transform, depth_transform


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CorruptedGestureDataset(Dataset):
    """
    Dataset with corruption labels
    Loads from: data/{clean, depth_occluded, low_light}
    """
    
    def __init__(self, root_dir, rgb_transform=None, depth_transform=None):
        """
        Args:
            root_dir: Path to data/ directory
        """
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        
        # Gesture classes
        self.classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Corruption types
        self.corruption_types = ['clean', 'depth_occluded', 'low_light']
        self.corruption_to_idx = {
            'clean': [0.0, 0.0],           # No corruption
            'depth_occluded': [0.0, 1.0],  # Depth corrupted
            'low_light': [1.0, 0.0],       # RGB corrupted
        }
        
        # Load all samples
        print(f"Loading corrupted dataset from {root_dir}...")
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_samples(self):
        """Load all samples with corruption labels"""
        samples = []
        
        for corruption_type in self.corruption_types:
            corruption_dir = os.path.join(self.root_dir, corruption_type)
            
            if not os.path.exists(corruption_dir):
                print(f"⚠️  {corruption_dir} not found, skipping...")
                continue
            
            for class_name in self.classes:
                class_dir = os.path.join(corruption_dir, class_name)
                
                if not os.path.exists(class_dir):
                    continue
                
                # Get all color images and sort
                color_files = sorted([f for f in os.listdir(class_dir) 
                                     if f.startswith('color_image_')])
                
                # Limit depth_occluded to first 20 samples per class
                if corruption_type == 'depth_occluded':
                    color_files = color_files[:20]
                
                for color_file in color_files:
                    sample_id = color_file.replace('color_image_', '').replace('.png', '')
                    
                    color_path = os.path.join(class_dir, color_file)
                    depth_path = os.path.join(class_dir, f'depth_image_{sample_id}.png')
                    
                    if os.path.exists(color_path) and os.path.exists(depth_path):
                        samples.append({
                            'color_path': color_path,
                            'depth_path': depth_path,
                            'label': self.class_to_idx[class_name],
                            'corruption_type': corruption_type,
                            'corruption_vector': self.corruption_to_idx[corruption_type],
                            'class_name': class_name
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            data: dict with 'rgb', 'depth'
            label: class label
            corruption: [rgb_corrupt, depth_corrupt] (0 or 1)
        """
        sample = self.samples[idx]
        
        # Load images
        rgb = Image.open(sample['color_path']).convert('RGB')
        depth = Image.open(sample['depth_path']).convert('L')
        
        # Apply transforms
        if self.rgb_transform:
            rgb = self.rgb_transform(rgb)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        
        data = {'rgb': rgb, 'depth': depth}
        label = sample['label']
        corruption = torch.tensor(sample['corruption_vector'], dtype=torch.float32)
        
        return data, label, corruption


def get_corrupted_dataloaders(data_dir, batch_size=16, num_workers=0, seed=42):
    """
    Create dataloaders for corrupted gesture dataset
    MODIFIED: Use 80/20 split (same as Stage 1) instead of 70/15/15
    """
    # Create dataset
    dataset = CorruptedGestureDataset(
        root_dir=data_dir,
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    print(f"Loading corrupted dataset from {data_dir}...")
    print(f"Loaded {len(dataset)} samples")
    
    # MODIFIED: 80/20 split (same as Stage 1 for consistency)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 192 samples
    val_size = total_size - train_size  # 48 samples
    
    # MODIFIED: Only 2-way split (no separate test set)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Split (80/20, matching Stage 1):")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples (used as test)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # MODIFIED: Return val_loader twice (serves as both val and test)
    return train_loader, val_loader, val_loader


def compute_allocation_loss(layer_allocation, corruption_labels):
    """
    Encourage correct layer allocation based on corruption
    
    Args:
        layer_allocation: [batch, 2, 12] - actual allocation
        corruption_labels: [batch, 2] - [rgb_corrupt, depth_corrupt]
    
    Returns:
        loss: scalar
    """
    batch_size = layer_allocation.size(0)
    
    # Count allocated layers per modality
    rgb_layers = layer_allocation[:, 0, :].sum(dim=1)    # [batch]
    depth_layers = layer_allocation[:, 1, :].sum(dim=1)  # [batch]
    
    # Corruption labels: [rgb_corrupt, depth_corrupt]
    rgb_corrupt = corruption_labels[:, 0]    # [batch]
    depth_corrupt = corruption_labels[:, 1]  # [batch]
    
    # Target allocation:
    # - If RGB corrupted: allocate more to Depth
    # - If Depth corrupted: allocate more to RGB
    # - If clean: allocate equally
    
    # Compute ideal ratio (higher corrupt → lower allocation)
    # Use softmax to get allocation ratio
    corruption_inv = 1.0 - corruption_labels  # Inverse corruption
    
    # Add small epsilon to avoid division by zero
    epsilon = 0.1
    corruption_inv = corruption_inv + epsilon
    
    # Normalize to get target ratio
    target_ratio = corruption_inv / corruption_inv.sum(dim=1, keepdim=True)
    target_rgb_ratio = target_ratio[:, 0]    # [batch]
    target_depth_ratio = target_ratio[:, 1]  # [batch]
    
    # Actual ratio
    total_layers = rgb_layers + depth_layers + 1e-6
    actual_rgb_ratio = rgb_layers / total_layers
    actual_depth_ratio = depth_layers / total_layers
    
    # MSE loss between target and actual ratios
    loss = nn.MSELoss()(actual_rgb_ratio, target_rgb_ratio) + \
           nn.MSELoss()(actual_depth_ratio, target_depth_ratio)
    
    return loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                alpha=1.0, beta=0.5, temperature=1.0):
    """
    Train for one epoch
    
    Loss = alpha * classification_loss + beta * allocation_loss
    """
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_alloc_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, labels, corruption) in enumerate(pbar):
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)
        labels = labels.to(device)
        corruption = corruption.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, layer_allocation = model(
            rgb, depth, 
            temperature=temperature, 
            return_allocation=True
        )
        
        # Classification loss
        cls_loss = criterion(logits, labels)
        
        # Allocation loss (encourage corruption-aware allocation)
        alloc_loss = compute_allocation_loss(layer_allocation, corruption)
        
        # Combined loss
        loss = alpha * cls_loss + beta * alloc_loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_alloc_loss += alloc_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'alloc': f'{alloc_loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_alloc_loss = total_alloc_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_cls_loss, avg_alloc_loss, accuracy


def validate(model, dataloader, criterion, device, temperature=1.0):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_alloc_loss = 0
    correct = 0
    total = 0
    
    # Track allocation statistics
    allocation_stats = {
        'clean': {'rgb': [], 'depth': []},
        'depth_occluded': {'rgb': [], 'depth': []},
        'low_light': {'rgb': [], 'depth': []}
    }
    
    with torch.no_grad():
        for data, labels, corruption in tqdm(dataloader, desc="Validating"):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            labels = labels.to(device)
            corruption = corruption.to(device)
            
            logits, layer_allocation = model(
                rgb, depth,
                temperature=temperature,
                return_allocation=True
            )
            
            cls_loss = criterion(logits, labels)
            alloc_loss = compute_allocation_loss(layer_allocation, corruption)
            loss = cls_loss + 0.5 * alloc_loss
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_alloc_loss += alloc_loss.item()
            
            # Record allocation stats
            for i in range(rgb.size(0)):
                rgb_count = layer_allocation[i, 0].sum().item()
                depth_count = layer_allocation[i, 1].sum().item()
                
                # Determine corruption type
                if corruption[i, 0] == 1.0:  # RGB corrupted
                    corr_type = 'low_light'
                elif corruption[i, 1] == 1.0:  # Depth corrupted
                    corr_type = 'depth_occluded'
                else:
                    corr_type = 'clean'
                
                allocation_stats[corr_type]['rgb'].append(rgb_count)
                allocation_stats[corr_type]['depth'].append(depth_count)
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_alloc_loss = total_alloc_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Compute average allocations
    avg_allocations = {}
    for corr_type in allocation_stats:
        if len(allocation_stats[corr_type]['rgb']) > 0:
            avg_allocations[corr_type] = {
                'rgb': np.mean(allocation_stats[corr_type]['rgb']),
                'depth': np.mean(allocation_stats[corr_type]['depth'])
            }
    
    return avg_loss, avg_cls_loss, avg_alloc_loss, accuracy, avg_allocations


def main(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading corrupted data from {args.data_dir}...")
    train_loader, val_loader, test_loader = get_corrupted_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating adaptive model...")
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=args.total_layers,
        qoi_dim=128,
        stage1_checkpoint=args.stage1_checkpoint
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer (only controller parameters)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    best_val_acc = 0
    
    print(f"\nStarting Stage 2 training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        # Anneal temperature (start from 1.0, decrease to 0.5)
        temperature = max(0.5, 1.0 - (epoch / args.epochs) * 0.5)
        
        # Train
        train_loss, train_cls, train_alloc, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            alpha=args.alpha, beta=args.beta, temperature=temperature
        )
        
        # Validate
        val_loss, val_cls, val_alloc, val_acc, allocations = validate(
            model, val_loader, criterion, device, temperature
        )
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/train_cls', train_cls, epoch)
        writer.add_scalar('Loss/train_alloc', train_alloc, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Temperature', temperature, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f} | Cls: {train_cls:.4f} | Alloc: {train_alloc:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f} | Cls: {val_cls:.4f} | Alloc: {val_alloc:.4f} | Acc: {val_acc:.2f}%")
        print(f"  Temperature: {temperature:.2f}")
        
        # Print allocation stats
        print(f"  Avg Allocations:")
        for corr_type, alloc in allocations.items():
            print(f"    {corr_type:15s}: RGB {alloc['rgb']:.1f} | Depth {alloc['depth']:.1f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, f'best_controller_{args.total_layers}layers.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'allocations': allocations
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print("="*60)
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model
    best_checkpoint = torch.load(
        os.path.join(args.output_dir, f'best_controller_{args.total_layers}layers.pth'),
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_cls, test_alloc, test_acc, test_allocations = validate(
        model, test_loader, criterion, device, temperature=0.5
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f} | Cls: {test_cls:.4f} | Alloc: {test_alloc:.4f} | Acc: {test_acc:.2f}%")
    print(f"  Allocations:")
    for corr_type, alloc in test_allocations.items():
        print(f"    {corr_type:15s}: RGB {alloc['rgb']:.1f} | Depth {alloc['depth']:.1f}")
    
    print("="*60)
    
    print(f"\n✅ Stage 2 training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Models saved to {args.output_dir}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 2 Adaptive Controller')
    
    parser.add_argument('--data_dir', type=str, 
                        default='data',
                        help='Path to corrupted data directory')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='checkpoints/stage1/best_model.pth',
                        help='Path to Stage 1 checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--total_layers', type=int, default=8,
                        help='Total layer budget for controller')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for allocation loss')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage2',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    main(args)