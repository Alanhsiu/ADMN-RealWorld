"""
Detailed evaluation answering TA's questions
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from scripts.train_stage2 import CorruptedGestureDataset, get_corrupted_dataloaders
from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import rgb_transform, depth_transform
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

device = torch.device("cuda")

# Load model
model = AdaptiveGestureClassifier(
    num_classes=4,
    adapter_hidden_dim=256,
    total_layers=12,
    qoi_dim=128,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
).to(device)

checkpoint = torch.load('checkpoints/stage2_L12/best_controller_12layers.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("="*80)
print("DETAILED EVALUATION - ANSWERING TA QUESTIONS")
print("="*80)

# Load full dataset to split by corruption type
dataset = CorruptedGestureDataset('data', rgb_transform, depth_transform)

# Separate samples by corruption type
clean_samples = [s for s in dataset.samples if s['corruption_type'] == 'clean']
depth_corrupted = [s for s in dataset.samples if s['corruption_type'] == 'depth_occluded']
low_light = [s for s in dataset.samples if s['corruption_type'] == 'low_light']

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

def evaluate_samples(samples, corruption_name):
    """Evaluate on specific corruption type"""
    all_labels = []
    all_preds = []
    all_allocations = []
    
    print(f"\n{'='*80}")
    print(f"Evaluating on {corruption_name.upper()} data ({len(samples)} samples)")
    print('='*80)
    
    with torch.no_grad():
        for sample in tqdm(samples, desc=corruption_name):
            # Load image
            from PIL import Image
            rgb = Image.open(sample['color_path']).convert('RGB')
            depth = Image.open(sample['depth_path']).convert('L')
            
            rgb = rgb_transform(rgb).unsqueeze(0).to(device)
            depth = depth_transform(depth).unsqueeze(0).to(device)
            
            # Forward
            logits, allocation = model(rgb, depth, temperature=1.0, return_allocation=True)
            pred = logits.argmax(1).item()
            
            all_labels.append(sample['label'])
            all_preds.append(pred)
            all_allocations.append(allocation[0].cpu().numpy())
    
    # Accuracy
    accuracy = 100 * np.mean(np.array(all_labels) == np.array(all_preds))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=class_names, 
                                   zero_division=0)
    
    # Average allocation
    avg_alloc = np.mean(all_allocations, axis=0)
    
    # Print results
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"\nAverage Allocation:")
    print(f"  RGB layers:   {avg_alloc[0].sum():.1f} / 12")
    print(f"  Depth layers: {avg_alloc[1].sum():.1f} / 12")
    print(f"  First layer always on: RGB[0]={avg_alloc[0,0]:.1f}, Depth[0]={avg_alloc[1,0]:.1f}")
    
    print(f"\nConfusion Matrix:")
    print("              " + "  ".join([f"{c:>10s}" for c in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>12s}  " + "  ".join([f"{x:>10d}" for x in row]))
    
    print(f"\nPer-Class Performance:")
    print(report)
    
    # Identify common errors
    print(f"\nCommon Errors:")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                print(f"  {class_names[i]} → {class_names[j]}: {cm[i, j]} times")
    
    return accuracy, cm, avg_alloc

# ==========================================
# Question 2: Accuracy on clean data
# ==========================================
clean_acc, clean_cm, clean_alloc = evaluate_samples(clean_samples, "CLEAN")

# ==========================================
# Corrupted data
# ==========================================
depth_acc, depth_cm, depth_alloc = evaluate_samples(depth_corrupted, "DEPTH OCCLUDED")
low_acc, low_cm, low_alloc = evaluate_samples(low_light, "LOW LIGHT")

# ==========================================
# Summary
# ==========================================
print("\n" + "="*80)
print("SUMMARY - ANSWERS TO TA QUESTIONS")
print("="*80)

print("\n[Q1] First layer activation:")
print(f"  RGB first layer:   {clean_alloc[0,0]:.1f} (should be 1.0)")
print(f"  Depth first layer: {clean_alloc[1,0]:.1f} (should be 1.0)")
if clean_alloc[0,0] < 0.9 or clean_alloc[1,0] < 0.9:
    print("  ⚠️  First layers NOT always activated - need to modify code")
else:
    print("  ✅ First layers always activated")

print("\n[Q2] Accuracy on clean data:")
print(f"  Clean accuracy: {clean_acc:.2f}%")
if clean_acc < 90:
    print(f"  ⚠️  Lower than expected for simple gestures")
    print("  Possible reasons:")
    print("    - Model optimized for corrupted data trade-off")
    print("    - Need more clean training data")
    print("    - Consider adjusting alpha/beta weights")
else:
    print("  ✅ Good performance on clean data")

print("\n[Q3] Error analysis:")
print(f"  Overall test accuracy: {(clean_acc*len(clean_samples) + depth_acc*len(depth_corrupted) + low_acc*len(low_light)) / (len(clean_samples)+len(depth_corrupted)+len(low_light)):.2f}%")
print("  See confusion matrices above for detailed error patterns")

print("\n[Q4] Corruption supervision:")
print(f"  Loss = alpha * classification + beta * allocation")
print(f"  alpha = 1.0  (classification loss weight)")
print(f"  beta = 5.0   (allocation loss weight)")
print(f"  Corruption labels: clean [0,0], depth_occluded [0,1], low_light [1,0]")
print(f"  Allocation loss encourages:")
print(f"    - RGB corrupted → allocate more to Depth")
print(f"    - Depth corrupted → allocate more to RGB")
print("  ✅ Yes, we use corruption supervision as auxiliary loss")

print("\n[Q5] Adaptive allocation behavior:")
print(f"  Clean:          RGB {clean_alloc[0].sum():.1f} | Depth {clean_alloc[1].sum():.1f}")
print(f"  Depth occluded: RGB {depth_alloc[0].sum():.1f} | Depth {depth_alloc[1].sum():.1f}")
print(f"  Low light:      RGB {low_alloc[0].sum():.1f} | Depth {low_alloc[1].sum():.1f}")

print("\n" + "="*80)