"""
Verify if first layer constraint is working
"""

import sys
sys.path.append('.')

import torch
from models.adaptive_controller import AdaptiveGestureClassifier

device = torch.device("cuda")

# Create model
model = AdaptiveGestureClassifier(
    num_classes=4,
    adapter_hidden_dim=256,
    total_layers=12,
    qoi_dim=128,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
).to(device)

print("="*70)
print("TESTING FIRST LAYER CONSTRAINT")
print("="*70)

# Test with random input
batch_size = 8
rgb = torch.randn(batch_size, 3, 224, 224).to(device)
depth = torch.randn(batch_size, 3, 224, 224).to(device)

model.eval()
with torch.no_grad():
    logits, allocation = model(rgb, depth, temperature=1.0, return_allocation=True)

print(f"\nAllocation shape: {allocation.shape}")  # Should be [8, 2, 12]

# Check first layers
rgb_first = allocation[:, 0, 0]   # RGB first layer [batch]
depth_first = allocation[:, 1, 0]  # Depth first layer [batch]

print(f"\nRGB first layer (should all be 1.0):")
print(f"  {rgb_first.cpu().numpy()}")
print(f"  Min: {rgb_first.min().item():.2f}, Max: {rgb_first.max().item():.2f}, Mean: {rgb_first.mean().item():.2f}")

print(f"\nDepth first layer (should all be 1.0):")
print(f"  {depth_first.cpu().numpy()}")
print(f"  Min: {depth_first.min().item():.2f}, Max: {depth_first.max().item():.2f}, Mean: {depth_first.mean().item():.2f}")

# Check total allocation
rgb_total = allocation[:, 0, :].sum(dim=1)
depth_total = allocation[:, 1, :].sum(dim=1)

print(f"\nTotal layers allocated:")
print(f"  RGB:   {rgb_total.cpu().numpy()}")
print(f"  Depth: {depth_total.cpu().numpy()}")
print(f"  Total: {(rgb_total + depth_total).cpu().numpy()}")
print(f"  Expected: 12 for all samples")

print("\n" + "="*70)

if rgb_first.min() >= 0.99 and depth_first.min() >= 0.99:
    print("✅ First layer constraint is WORKING!")
else:
    print("❌ First layer constraint is NOT working!")
    print("\nPossible issues:")
    print("1. Code modification didn't take effect")
    print("2. Old model was loaded (not retrained)")
    print("3. Bug in the forward() implementation")

print("="*70)