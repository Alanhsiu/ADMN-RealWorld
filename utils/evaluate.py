import torch

best = torch.load('checkpoints/stage1/best_model.pth', map_location='cpu')
final = torch.load('checkpoints/stage1/final_model.pth', map_location='cpu')

print("="*60)
print("TRAINING RESULTS")
print("="*60)
print(f"\nBest Model (Epoch {best['epoch']}):")
print(f"  Val Accuracy: {best['val_acc']:.2f}%")
print(f"  Val Loss: {best['val_loss']:.4f}")

print(f"\nFinal Model (Epoch {final['epoch']}):")
print(f"  Test Accuracy: {final['test_acc']:.2f}%")
print(f"  Test Loss: {final['test_loss']:.4f}")
print("="*60)