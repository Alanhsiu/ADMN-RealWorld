import os
import sys
from time import time
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

    with torch.no_grad(): # no need to track gradients (backward)
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)

        # Forward pass
        logits,RGB_back, depth_back, fusion = model(rgb, depth)

        # Predictions
        _, predicted = torch.max(logits, 1)
        # print([class_names[i] for i in predicted.cpu().numpy()])
    return predicted, RGB_back, depth_back, fusion

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    print("\nCreating model...")
    model = GestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        vision_vit_layers=12,
        depth_vit_layers=12,
        pretrained_path=args.pretrained_path,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    di = ["standing", "left_hand", "right_hand", "both_hands"]
    reason = ["image", "RGB_back", "depth_back", "fusion", "all"]
    for i in range(4):
        correct = 0
        total_time = 0
        avg_time = [0, 0, 0, 0, 0]
        for k in range(1, 21):
            st1 = time()
            data = single_image_transform(os.path.join(args.data_dir, di[i], f'color_image_{k:04d}.png'), os.path.join(args.data_dir, di[i], f'depth_image_{k:04d}.png'))
            st2 = time()
            # inference
            predicate, t1, t2, t3=inference(model, data, device)
            st3 = time()
            correct += predicate == i
            avg_time[0]+=st2-st1
            avg_time[1]+=t1
            avg_time[2]+=t2
            avg_time[3]+=t3
            avg_time[4]+=st3-st1
        for k in range(5):
            print(f'Avg inference time for {reason[k]} forclass {di[i]}: {avg_time[k]/20} seconds')
        accuracy = 100 * correct / 20
        print(f'Class: {di[i]}, Accuracy: {accuracy}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 1 Gesture Inference Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../data/simple_raw_data',
                        help='Path to processed data')
    
    # Model
    parser.add_argument('--pretrained_path', type=str,
                        default='checkpoints/pretrained/MAE_Dropout_FT_Dropout.pth',
                        help='Path to pretrained MAE weights')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Path to model checkpoint for inference')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)
    
    # use tensorboard to visualize the training process
    # tensorboard --logdir checkpoints/stage1/logs --port 6006 --bind_all
