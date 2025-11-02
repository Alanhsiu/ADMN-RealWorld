"""
Gesture Classification Model (based on GTDM_Early)
Modified from localization to classification
"""

import torch
import torch.nn as nn
import sys
import os
import time
# Add GTDM path to import their modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../GTDM_Lowlight'))

from models.timm_vit import VisionTransformer
from models.vit_dev import TransformerEnc, positionalencoding1d


class Adapter(nn.Module):
    """
    Adapter to project backbone features to fusion dimension
    (Copied from GTDM adapters.py)
    """
    def __init__(self, input_dim=768, output_dim=256):
        super(Adapter, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.fc(x))


class GestureClassifier(nn.Module):
    """
    RGB-D Gesture Classifier (Modified from GTDM_Early)
    
    Changes from GTDM:
    1. Output: 4-class classification instead of (x, y) localization
    2. Loss: CrossEntropy instead of MSE
    3. Single RGB-Depth pair (no distributed nodes)
    """
    
    def __init__(self, num_classes=4, adapter_hidden_dim=256, 
                 vision_vit_layers=12, depth_vit_layers=12,
                 pretrained_path=None, layerdrop=0.0):
        super(GestureClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Fusion transformer parameters (same as GTDM)
        dim_dec = 256
        depth_dec = 6
        heads = 4
        
        # RGB ViT backbone (same as GTDM vision)
        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=vision_vit_layers, 
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=layerdrop
        )
        
        # Depth ViT backbone (same as GTDM depth)
        self.depth = VisionTransformer(
            patch_size=16, embed_dim=768, depth=depth_vit_layers,
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=layerdrop
        )
        
        # Load pretrained MAE weights if using 12 layers
        if vision_vit_layers == 12 and pretrained_path:
            self._load_pretrained_weights(pretrained_path)
        
        # Adapters to project to common dimension (same as GTDM)
        self.vision_adapter = Adapter(768, adapter_hidden_dim)
        self.depth_adapter = Adapter(768, adapter_hidden_dim)
        
        # Multimodal fusion transformer (same as GTDM encoder)
        self.encoder = TransformerEnc(
            dim=dim_dec, depth=depth_dec, heads=heads,
            dim_head=dim_dec//heads, mlp_dim=3*dim_dec
        )
        
        # Classification head (CHANGED: from localization to classification)
        self.classifier = nn.Sequential(
            nn.Linear(dim_dec, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def _load_pretrained_weights(self, pretrained_path):
        """Load MAE pretrained weights and freeze early layers"""
        print(f"Loading pretrained weights from {pretrained_path}")
        
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Load for vision backbone
            vision_msg = self.vision.load_state_dict(checkpoint['model'], strict=False)
            print(f"Vision backbone: {vision_msg}")
            
            # Load for depth backbone
            depth_msg = self.depth.load_state_dict(checkpoint['model'], strict=False)
            print(f"Depth backbone: {depth_msg}")
            
            # Freeze early layers (same strategy as GTDM)
            # Vision: freeze all except last layer
            for param in self.vision.parameters():
                param.requires_grad = False
            for param in self.vision.blocks[11].parameters():
                param.requires_grad = True
            
            # Depth: freeze all except last 2 layers
            for param in self.depth.parameters():
                param.requires_grad = False
            for param in self.depth.blocks[10].parameters():
                param.requires_grad = True
            for param in self.depth.blocks[11].parameters():
                param.requires_grad = True
            
            print("✅ Pretrained weights loaded and early layers frozen")
        else:
            print(f"⚠️  Pretrained weights not found at {pretrained_path}")
            print("Training from scratch...")
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: [batch_size, 3, 224, 224]
            depth: [batch_size, 3, 224, 224]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        outlist = []
        
        # During training, use layerdrop (same as GTDM)
        if self.training:
            dropped_layers_img = (torch.rand(12) > self.vision.layerdrop_rate).int().to(rgb.device)
            dropped_layers_depth = (torch.rand(12) > self.depth.layerdrop_rate).int().to(depth.device)
            
            # Full modality dropout (10% chance)
            if torch.rand(1).item() < 0.1:
                dropped_layers_img = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(rgb.device)
            if torch.rand(1).item() < 0.1:
                dropped_layers_depth = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(depth.device)
        else:
            # No dropout during evaluation
            dropped_layers_img = torch.ones(12).int().to(rgb.device)
            dropped_layers_depth = torch.ones(12).int().to(depth.device)
        
        # Extract RGB features
        t0 = time.time()
        rgb_features = self.vision.forward_train(rgb, dropped_layers_img)
        rgb_features = torch.squeeze(rgb_features)
        if len(rgb_features.shape) == 1:
            rgb_features = torch.unsqueeze(rgb_features, dim=0)
        outlist.append(self.vision_adapter(rgb_features))
        
        # Extract Depth features
        t1 = time.time()
        depth_features = self.depth.forward_train(depth, dropped_layers_depth)
        depth_features = torch.squeeze(depth_features)
        if len(depth_features.shape) == 1:
            depth_features = torch.unsqueeze(depth_features, dim=0)
        outlist.append(self.depth_adapter(depth_features))
        
        t2 = time.time()
        # Stack features: [batch_size, 2, dim]
        agg_features = torch.stack(outlist, dim=1)
        
        # Add positional encoding
        b, n, d = agg_features.shape
        agg_features += positionalencoding1d(d, n)
        
        # Multimodal fusion
        fused = self.encoder(agg_features)  # [batch_size, 2, 256]
        
        # Mean pooling
        fused = torch.mean(fused, dim=1)  # [batch_size, 256]
        
        # Classification (CHANGED: from localization to classification)
        logits = self.classifier(fused)  # [batch_size, num_classes]
        t3 = time.time()
        return logits, t1-t0, t2-t1, t3-t2


# Test the model
if __name__ == "__main__":
    print("Testing GestureClassifier...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = GestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        vision_vit_layers=12,
        depth_vit_layers=12,
        pretrained_path='checkpoints/pretrained/MAE_Dropout_FT_Dropout.pth'
    ).to(device)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    depth = torch.randn(batch_size, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(rgb, depth)
    
    print(f"\nForward pass test:")
    print(f"  Input RGB: {rgb.shape} on {rgb.device}")
    print(f"  Input Depth: {depth.shape} on {depth.device}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Predictions: {torch.argmax(logits, dim=1)}")
    
    print("\n✅ Model test passed!")
