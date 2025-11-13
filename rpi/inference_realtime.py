import os
import sys

from sklearn import pipeline
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import numpy as np
import pyrealsense2 as rs
import cv2
import time
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.adaptive_controller import AdaptiveGestureClassifier
from data.gesture_dataset import transform_from_camera

class_names = ['standing', 'left_hand', 'right_hand', 'both_hands']

def conceptual_calculate_flops(total_layers, rgb_layers_used, depth_layers_used):
    """Calculates FLOPs based on utilized layers (Conceptual Estimate)."""
    
    BASE_FLOPS = 10.0e9    # Fixed cost (GFLOPs)
    PER_LAYER_FLOPS = 0.5e9 # Cost per ViT layer (GFLOPs)
    
    total_flops = BASE_FLOPS + \
                  (rgb_layers_used * PER_LAYER_FLOPS) + \
                  (depth_layers_used * PER_LAYER_FLOPS)
                  
    return total_flops / 1e9 # Return in GFLOPs

def inference(model, data, device):
    """Validate the model"""

    with torch.no_grad(): # no need to track gradients (backward)
        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)

        # Forward pass
        logits, layer_allocation = model(
            rgb, depth, 
            temperature=1.0, 
            return_allocation=True
        )

        # Predictions
        _, predicted = torch.max(logits, 1)
        rgb_layers = layer_allocation[0, 0, :].sum().item()
        depth_layers = layer_allocation[0, 1, :].sum().item()
        print("===> Used Layers - RGB:", int(rgb_layers), "Depth:", int(depth_layers))
        # print(logits)
        print("===> Result:", [class_names[i] for i in predicted.cpu().numpy()])

    return  rgb_layers, depth_layers

def take_pic(model=None, device=None):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams (you can adjust resolution, format, and FPS as needed)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    lat = 0
    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
 
        if not color_frame or not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Colormap', depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        
        # if now - last_capture_time >= 5.0:
        if key == ord('q'):
            now = time.time()
            print("==> Captured one RGB+Depth, sending to model...")
        
            data = transform_from_camera(color_image, depth_colormap)
            rgb_layers, depth_layers = inference(model, data, device)
            flops = conceptual_calculate_flops(
                total_layers=args.total_layers,
                rgb_layers_used=rgb_layers,
                depth_layers_used=depth_layers
            )
            print(f"===> Estimated FLOPs for this inference: {flops:.2f} GFLOPs")
            print("===> Latency:", time.time() - now, "seconds")
            lat += time.time() - now
            print("================================================")

        elif key == 27:  # ESC
            # print("Exit.")
            pipeline.stop()
            cv2.destroyAllWindows()
            print("===> Avg Latency:", lat/10, "seconds")
            break
    
        



def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Create model
    print("\nCreating model...")
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=args.total_layers,
        qoi_dim=128,
        stage1_checkpoint=None,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False )

    # load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    take_pic(model, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 1 Gesture Inference Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../data/processed/train/left_hand',
                        help='Path to processed data')
    
    # Model
    parser.add_argument('--pretrained_path', type=str,
                        default='checkpoints/pretrained/MAE_Dropout_FT_Dropout.pth',
                        help='Path to pretrained MAE weights')
    parser.add_argument('--checkpoint', type=str, default='best_controller_12layers.pth',
                        help='Path to model checkpoint for inference')
    
    parser.add_argument('--total_layers', type=int, default=12,
                        help='Total layer budget for controller')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints/stage1',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)
    
    # use tensorboard to visualize the training process
    # tensorboard --logdir checkpoints/stage1/logs --port 6006 --bind_all
