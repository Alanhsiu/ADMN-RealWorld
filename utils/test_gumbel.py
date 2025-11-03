# test_gumbel.py
import torch
from models.adaptive_controller import AdaptiveGestureClassifier

model = AdaptiveGestureClassifier(
    num_classes=4,
    adapter_hidden_dim=256,
    total_layers=12,
    qoi_dim=128,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Create dummy input
rgb = torch.randn(2, 3, 224, 224).to(device)
depth = torch.randn(2, 3, 224, 224).to(device)

print("Testing allocation consistency (should have randomness with Gumbel):")
for i in range(3):
    with torch.no_grad():
        _, allocation = model(rgb, depth, temperature=1.0, return_allocation=True)
    print(f"Run {i+1}: RGB={allocation[0,0].sum():.1f}, Depth={allocation[0,1].sum():.1f}")