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
from data.gesture_dataset import single_image_transform

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

def inference(model, data, device):
    """Validate the model"""
    model.eval()

    with torch.no_grad(): # no need to track gradients (backward)
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)

        # Forward pass
        logits = model(rgb, depth)

        # Predictions
        _, predicted = torch.max(logits, 1)
        print([class_names[i] for i in predicted.cpu().numpy()])

    return

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    data = single_image_transform(os.path.join(args.data_dir, 'color_image_0003.png'), os.path.join(args.data_dir, 'depth_image_0003.png'))
    # Create model
    print("\nCreating model...")
    model = GestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        vision_vit_layers=12,
        depth_vit_layers=12,
        pretrained_path=args.pretrained_path,
    ).to(device)

    checkpoint = torch.load('checkpoints/stage1/best_model.pth', map_location='cpu')

    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # inference
    _=inference(model, data, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 1 Gesture Inference Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../data/processed/train/left_hand',
                        help='Path to processed data')
    
    # Model
    parser.add_argument('--pretrained_path', type=str,
                        default='checkpoints/pretrained/MAE_Dropout_FT_Dropout.pth',
                        help='Path to pretrained MAE weights')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/stage1/best_model.pth',
                        help='Path to model checkpoint for inference')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)
    
    # use tensorboard to visualize the training process
    # tensorboard --logdir checkpoints/stage1/logs --port 6006 --bind_all
