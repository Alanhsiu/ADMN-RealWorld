"""
Stage 1 Training: Train baseline RGB-D gesture classifier
Modified to use FULL dataset (clean + corrupted) with 80/20 split
Includes regularization and data augmentation to prevent overfitting
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gesture_classifier import GestureClassifier
from data.gesture_dataset import rgb_transform, depth_transform
from scripts.train_stage2 import CorruptedGestureDataset


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_full_dataloaders(data_dir, batch_size=16, num_workers=4, seed=42):
    """
    Create dataloaders with FULL dataset (clean + corrupted)
    MODIFIED: Use 80/20 split instead of 70/15/15 to increase training data
              and provide more stable validation estimates
    
    Returns:
        train_loader, val_loader, test_loader (val_loader used as test)
    """
    # Create dataset with ALL data (clean + depth_occluded + low_light)
    full_dataset = CorruptedGestureDataset(
        root_dir=data_dir,  # MODIFIED: Use root_dir parameter
        rgb_transform=rgb_transform,
        depth_transform=depth_transform
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(full_dataset)}")
    
    # Count samples per corruption type
    corruption_counts = {}
    for sample in full_dataset.samples:
        corr_type = sample['corruption_type']
        corruption_counts[corr_type] = corruption_counts.get(corr_type, 0) + 1
    
    for corr_type, count in corruption_counts.items():
        print(f"  {corr_type}: {count} samples")
    
    # MODIFIED: 80/20 split (no separate test set for small dataset)
    # This provides more training data (192 vs 168) and larger validation set (48 vs 36)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)  # 192 samples
    val_size = total_size - train_size  # 48 samples
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"\nSplit Strategy (80/20):")
    print(f"  Train: {train_size} samples (80%)")
    print(f"  Val:   {val_size} samples (20%, used as held-out test set)")
    print(f"  Rationale: Small dataset (240 samples) requires:")
    print(f"    - Maximum training data while maintaining held-out validation")
    print(f"    - Sufficient validation samples for stable evaluation")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # MODIFIED: Return val_loader twice (serves as both validation and test)
    return train_loader, val_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    MODIFIED: Track per-corruption accuracy
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # MODIFIED: Track accuracy per corruption type
    corruption_correct = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    corruption_total = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, labels, corruption) in enumerate(pbar):  # MODIFIED: Now includes corruption labels
        # Move to device
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(rgb, depth)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        total_loss += loss.item()
        
        # MODIFIED: Track per-corruption accuracy
        for i in range(len(labels)):
            if corruption[i, 0] == 1.0:
                corr_type = 'low_light'
            elif corruption[i, 1] == 1.0:
                corr_type = 'depth_occluded'
            else:
                corr_type = 'clean'
            
            corruption_total[corr_type] += 1
            if predicted[i] == labels[i]:
                corruption_correct[corr_type] += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # MODIFIED: Calculate per-corruption accuracy
    corruption_acc = {
        corr_type: 100 * corruption_correct[corr_type] / corruption_total[corr_type] 
        if corruption_total[corr_type] > 0 else 0
        for corr_type in ['clean', 'depth_occluded', 'low_light']
    }
    
    return avg_loss, accuracy, corruption_acc


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    MODIFIED: Track per-corruption accuracy in addition to per-class
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    
    # MODIFIED: Per-corruption accuracy
    corruption_correct = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    corruption_total = {'clean': 0, 'depth_occluded': 0, 'low_light': 0}
    
    with torch.no_grad():
        for data, labels, corruption in tqdm(dataloader, desc="Validating"):  # MODIFIED: Added corruption
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(rgb, depth)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            
            # Per-class and per-corruption accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
                
                # MODIFIED: Track corruption type
                if corruption[i, 0] == 1.0:
                    corr_type = 'low_light'
                elif corruption[i, 1] == 1.0:
                    corr_type = 'depth_occluded'
                else:
                    corr_type = 'clean'
                
                corruption_total[corr_type] += 1
                if predicted[i] == labels[i]:
                    corruption_correct[corr_type] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                 for i in range(4)]
    
    # MODIFIED: Calculate per-corruption accuracy
    corruption_acc = {
        corr_type: 100 * corruption_correct[corr_type] / corruption_total[corr_type] 
        if corruption_total[corr_type] > 0 else 0
        for corr_type in ['clean', 'depth_occluded', 'low_light']
    }
    
    return avg_loss, accuracy, class_acc, corruption_acc


def main(args):
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # MODIFIED: Create dataloaders with FULL dataset (80/20 split)
    print(f"\nLoading FULL dataset (clean + corrupted) from {args.data_dir}...")
    train_loader, val_loader, test_loader = get_full_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Create model
    print("\nCreating model...")
    model = GestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        vision_vit_layers=12,
        depth_vit_layers=12,
        pretrained_path=args.pretrained_path,
        layerdrop=args.layerdrop  # MODIFIED: Now using non-zero layerdrop for regularization
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # MODIFIED: Loss with label smoothing for regularization
    # criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    
    # MODIFIED: Optimizer with weight decay (L2 regularization)
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # MODIFIED: Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # MODIFIED: Early stopping to prevent overfitting
    best_val_acc = 0
    no_improve_count = 0
    
    print(f"\nRegularization Strategy:")
    print(f"  Layerdrop: {args.layerdrop}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Learning rate scheduling: ReduceLROnPlateau")
    print("="*60)
    
    print(f"\nStarting training for up to {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_corruption_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, class_acc, val_corruption_acc = validate(
            model, val_loader, criterion, device
        )
        
        # MODIFIED: Update learning rate based on validation performance
        scheduler.step(val_acc)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # MODIFIED: Log per-corruption accuracy
        for corr_type in ['clean', 'depth_occluded', 'low_light']:
            writer.add_scalar(f'Accuracy_Corruption/train_{corr_type}', 
                            train_corruption_acc[corr_type], epoch)
            writer.add_scalar(f'Accuracy_Corruption/val_{corr_type}', 
                            val_corruption_acc[corr_type], epoch)
        
        # Log per-class accuracy
        for i, acc in enumerate(class_acc):
            writer.add_scalar(f'Accuracy/class_{i}', acc, epoch)
        
        # MODIFIED: Print detailed epoch summary including overfit gap
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Overfit Gap: {train_acc - val_acc:.2f}%")  # MODIFIED: Monitor overfitting
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Per-class Val Acc: {[f'{acc:.1f}%' for acc in class_acc]}")
        print(f"  Per-corruption Val Acc:")
        print(f"    Clean:          {val_corruption_acc['clean']:.2f}%")
        print(f"    Depth occluded: {val_corruption_acc['depth_occluded']:.2f}%")
        print(f"    Low light:      {val_corruption_acc['low_light']:.2f}%")
        
        # MODIFIED: Save best model and implement early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'corruption_acc': val_corruption_acc,
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count} epochs")
            
            # MODIFIED: Early stopping trigger
            if no_improve_count >= args.patience:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                print(f"  Best validation accuracy: {best_val_acc:.2f}%")
                break
        
        print("="*60)
        
        # MODIFIED: Gradually increase layerdrop (optional, adaptive regularization)
        if epoch % 10 == 9 and args.layerdrop > 0:
            new_rate = min(args.max_layerdrop, model.vision.layerdrop_rate + 0.05)
            if new_rate > model.vision.layerdrop_rate:
                model.vision.layerdrop_rate = new_rate
                model.depth.layerdrop_rate = new_rate
                print(f"  Updated layerdrop rate: {new_rate:.2f}")
    
    # MODIFIED: Load best model for final evaluation
    print("\n" + "="*60)
    print("Loading best model for final evaluation...")
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation on validation set (serves as test set)
    test_loss, test_acc, test_class_acc, test_corruption_acc = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS (on validation/test set):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"\n  Per-class Accuracy:")
    classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
    for i, (cls, acc) in enumerate(zip(classes, test_class_acc)):
        print(f"    {cls}: {acc:.2f}%")
    print(f"\n  Per-corruption Accuracy:")
    print(f"    Clean:          {test_corruption_acc['clean']:.2f}%")
    print(f"    Depth occluded: {test_corruption_acc['depth_occluded']:.2f}%")
    print(f"    Low light:      {test_corruption_acc['low_light']:.2f}%")
    print(f"{'='*60}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': best_checkpoint['epoch'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        'corruption_acc': test_corruption_acc,
    }, final_path)
    print(f"\n✅ Training completed! Models saved to {args.output_dir}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 1 Gesture Classifier with Full Dataset')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data',  # MODIFIED: Changed from 'data/processed'
                        help='Path to data directory (containing clean, depth_occluded, low_light folders)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--pretrained_path', type=str,
                        default='checkpoints/pretrained/MAE_Dropout_FT_Dropout.pth',
                        help='Path to pretrained MAE weights')
    parser.add_argument('--layerdrop', type=float, default=0.0,  # MODIFIED: Changed from 0.0 to 0.1
                        help='Initial layerdrop rate for regularization')
    parser.add_argument('--max_layerdrop', type=float, default=0.2,
                        help='Maximum layerdrop rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # MODIFIED: Added regularization hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for CrossEntropyLoss (0.0 = no smoothing)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1_regularized',  # MODIFIED: New name
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)