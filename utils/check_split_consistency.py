import torch
from scripts.train_stage1 import get_full_dataloaders as get_stage1_loaders
from scripts.train_stage2 import get_corrupted_dataloaders as get_stage2_loaders

# Load both
_, val1, _ = get_stage1_loaders('data', batch_size=1, seed=42)
_, val2, _ = get_stage2_loaders('data', batch_size=1, seed=42)

# Get first sample from each
sample1 = next(iter(val1))
sample2 = next(iter(val2))

# Compare
data1, label1, corr1 = sample1
data2, label2, corr2 = sample2

print(f"Stage 1 val first sample: label={label1.item()}, corruption={corr1}")
print(f"Stage 2 val first sample: label={label2.item()}, corruption={corr2}")

if label1.item() == label2.item() and torch.equal(corr1, corr2):
    print("✅ Splits are consistent!")
else:
    print("❌ Splits are DIFFERENT!")