import torch
import random
import numpy as np
from models.adaptive_controller import AdaptiveGestureClassifier

# Set all seeds
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model = AdaptiveGestureClassifier(
    num_classes=4,
    adapter_hidden_dim=256,
    total_layers=12,
    qoi_dim=128,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
)

# Load checkpoint
checkpoint = torch.load('checkpoints/stage2/best_controller_12layers.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Force eval mode on all submodules
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

from scripts.train_stage2 import get_corrupted_dataloaders
import torch.nn as nn

_, val_loader, _ = get_corrupted_dataloaders('data', batch_size=48, seed=42)

# Run evaluation 3 times
print("\nRunning evaluation 3 times (should be identical):")

first_allocation = None

for i in range(3):
    print(f"\nRun {i+1}:")
    print(f"  model.training: {model.training}")
    print(f"  layer_allocator.training: {model.layer_allocator.training}")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, labels, corruption) in enumerate(val_loader):
            rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            labels = labels.to(device)
            
            logits, allocation = model(rgb, depth, temperature=0.5, return_allocation=True)
            
            # MODIFIED: Check if allocation is deterministic (only first batch)
            if batch_idx == 0:
                if i == 0:
                    first_allocation = allocation[0].cpu().clone()
                    print(f"  First batch allocation saved:")
                    print(f"    RGB:   {allocation[0, 0]}")
                    print(f"    Depth: {allocation[0, 1]}")
                else:
                    current_allocation = allocation[0].cpu()
                    if torch.equal(current_allocation, first_allocation):
                        print(f"  ✅ Allocation matches Run 1")
                    else:
                        print(f"  ❌ Allocation DIFFERS from Run 1!")
                        print(f"    RGB:   {allocation[0, 0]}")
                        print(f"    Depth: {allocation[0, 1]}")
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"  Accuracy: {accuracy:.2f}%")

print("\n" + "="*60)
if first_allocation is not None:
    print("If all runs show:")
    print("  1. ✅ Allocation matches")
    print("  2. Same accuracy")
    print("Then the fix worked! The model is now deterministic in eval mode.")
else:
    print("Could not verify allocation consistency.")
print("="*60)