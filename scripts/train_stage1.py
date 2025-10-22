"""
Stage 1 Training: Train baseline RGB-D gesture classifier
Modified from GTDM train.py
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
from data.gesture_dataset import get_dataloaders


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, labels) in enumerate(pbar):
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
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Validating"):
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
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                 for i in range(4)]
    
    return avg_loss, accuracy, class_acc


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
    
    # Create dataloaders
    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = GestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        vision_vit_layers=12,
        depth_vit_layers=12,
        pretrained_path=args.pretrained_path,
        layerdrop=args.layerdrop
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    best_val_acc = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        for i, acc in enumerate(class_acc):
            writer.add_scalar(f'Accuracy/class_{i}', acc, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Per-class Acc: {[f'{acc:.1f}%' for acc in class_acc]}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print("="*60)
        
        # Gradually increase layerdrop (same as GTDM)
        if epoch % 10 == 9 and args.layerdrop > 0:
            model.vision.layerdrop_rate = min(args.max_layerdrop, 
                                              model.vision.layerdrop_rate + 0.1)
            model.depth.layerdrop_rate = min(args.max_layerdrop,
                                             model.depth.layerdrop_rate + 0.1)
            print(f"Updated layerdrop rate: {model.vision.layerdrop_rate:.1f}")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_class_acc = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*60}")
    print("Final Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Per-class Accuracy:")
    classes = ['standing', 'left_hand', 'right_hand', 'both_hands']
    for i, (cls, acc) in enumerate(zip(classes, test_class_acc)):
        print(f"    {cls}: {acc:.2f}%")
    print(f"{'='*60}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, final_path)
    print(f"\n✅ Training completed! Models saved to {args.output_dir}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 1 Gesture Classifier')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--pretrained_path', type=str,
                        default='checkpoints/pretrained/MAE_Dropout_FT_Dropout.pth',
                        help='Path to pretrained MAE weights')
    parser.add_argument('--layerdrop', type=float, default=0.0,
                        help='Initial layerdrop rate')
    parser.add_argument('--max_layerdrop', type=float, default=0.2,
                        help='Maximum layerdrop rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)
    
    # use tensorboard to visualize the training process
    # tensorboard --logdir checkpoints/stage1/logs --port 6006 --bind_all
