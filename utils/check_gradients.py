# check_gradients.py
import torch
from models.adaptive_controller import AdaptiveGestureClassifier
import torch.nn as nn

model = AdaptiveGestureClassifier(
    num_classes=4,
    adapter_hidden_dim=256,
    total_layers=12,
    qoi_dim=128,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
)

device = torch.device('cuda')
model.to(device)
model.train()

# Create dummy input
rgb = torch.randn(4, 3, 224, 224).to(device)
depth = torch.randn(4, 3, 224, 224).to(device)
labels = torch.tensor([0, 1, 2, 3]).to(device)
corruption = torch.tensor([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=torch.float32).to(device)

# Forward
logits, allocation = model(rgb, depth, temperature=1.0, return_allocation=True)

# Compute loss
criterion = nn.CrossEntropyLoss()
cls_loss = criterion(logits, labels)

# Allocation loss (simplified)
alloc_loss = allocation.sum()

loss = cls_loss + 5.0 * alloc_loss

# Backward
loss.backward()

# Check gradients
print("Checking gradients:")
print(f"  QoI RGB conv: {model.qoi_perception.rgb_conv[0].weight.grad is not None}")
print(f"  QoI Depth conv: {model.qoi_perception.depth_conv[0].weight.grad is not None}")
print(f"  Layer allocator: {model.layer_allocator.allocation_net[0].weight.grad is not None}")

if model.qoi_perception.rgb_conv[0].weight.grad is not None:
    print(f"  QoI gradient norm: {model.qoi_perception.rgb_conv[0].weight.grad.norm():.4f}")
else:
    print("  ❌ No gradient flowing to QoI!")

if model.layer_allocator.allocation_net[0].weight.grad is not None:
    print(f"  Allocator gradient norm: {model.layer_allocator.allocation_net[0].weight.grad.norm():.4f}")
else:
    print("  ❌ No gradient flowing to Allocator!")