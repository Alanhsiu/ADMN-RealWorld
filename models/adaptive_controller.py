"""
Adaptive Controller for Stage 2
Dynamically allocates layers based on RGB-D quality
Based on GTDM Conv_GTDM_Controller
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../GTDM_Lowlight'))

from models.timm_vit import VisionTransformer
from models.vit_dev import TransformerEnc, positionalencoding1d


class Adapter(nn.Module):
    """Adapter to project features"""
    def __init__(self, input_dim=768, output_dim=256):
        super(Adapter, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.fc(x))


class QoIPerceptionModule(nn.Module):
    """
    Perceive Quality-of-Information (QoI) from downsampled inputs
    Uses lightweight CNNs to extract QoI features
    """
    def __init__(self, output_dim=128):
        super(QoIPerceptionModule, self).__init__()
        
        # RGB perception (3 channels)
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        )
        
        # Depth perception (3 channels, expanded from 1)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Project to output dimension
        self.rgb_proj = nn.Linear(128, output_dim)
        self.depth_proj = nn.Linear(128, output_dim)
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: [batch, 3, 224, 224]
            depth: [batch, 3, 224, 224]
        Returns:
            qoi_features: [batch, output_dim * 2] - concatenated RGB and Depth QoI
        """
        # Extract QoI features
        rgb_feat = self.rgb_conv(rgb)  # [batch, 128, 1, 1]
        rgb_feat = rgb_feat.squeeze(-1).squeeze(-1)  # [batch, 128]
        rgb_qoi = self.rgb_proj(rgb_feat)  # [batch, output_dim]
        
        depth_feat = self.depth_conv(depth)
        depth_feat = depth_feat.squeeze(-1).squeeze(-1)
        depth_qoi = self.depth_proj(depth_feat)
        
        # Concatenate for fusion
        qoi_features = torch.cat([rgb_qoi, depth_qoi], dim=-1)  # [batch, output_dim * 2]
        
        return qoi_features


class LayerAllocationModule(nn.Module):
    """
    Allocate layers based on QoI features
    Uses Gumbel-Softmax for differentiable discrete sampling
    """
    def __init__(self, qoi_dim=128, hidden_dim=256, total_layers=8):
        super(LayerAllocationModule, self).__init__()
        
        self.total_layers = total_layers
        
        # MLP to predict layer allocation logits
        # Input: qoi_dim * 2 (RGB QoI + Depth QoI)
        # Output: 24 (12 RGB layers + 12 Depth layers)
        self.allocation_net = nn.Sequential(
            nn.Linear(qoi_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 24)  # 12 RGB + 12 Depth
        )

    def forward(self, qoi_features, temperature=1.0):
        """
        Args:
            qoi_features: [batch, qoi_dim * 2]
            temperature: Temperature for Gumbel-Softmax
        
        Returns:
            allocation: [batch, 2, 12] - binary allocation (0 or 1)
        """
        batch_size = qoi_features.size(0)
        
        # Predict logits for all 24 layers (12 RGB + 12 Depth)
        logits = self.allocation_net(qoi_features)  # [batch, 24]
        logits = logits.view(batch_size, 2, 12)    # [batch, 2, 12]
        
        # Reserve first layers (always activated)
        allocation_full = torch.zeros_like(logits)
        allocation_full[:, 0, 0] = 1.0  # RGB first layer
        allocation_full[:, 1, 0] = 1.0  # Depth first layer
        
        # Select from remaining layers
        remaining_budget = self.total_layers - 2
        
        # Get selectable logits (exclude first layers)
        selectable_indices = []
        for i in range(24):
            if i != 0 and i != 12:  # Skip first layer of each modality
                selectable_indices.append(i)
        
        logits_flat = logits.view(batch_size, -1)
        selectable_logits = logits_flat[:, selectable_indices]  # [batch, 22]
        
        # Apply Gumbel-Softmax (with noise in both training and eval)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selectable_logits) + 1e-10) + 1e-10)
        logits_with_noise = (selectable_logits + gumbel_noise) / temperature
        soft_allocation = F.softmax(logits_with_noise, dim=-1)
        
        # Top-k selection (greedy)
        _, top_k_indices = torch.topk(soft_allocation, k=remaining_budget, dim=-1)
        
        # Create binary allocation (hard)
        allocation_binary_hard = torch.zeros_like(selectable_logits)
        allocation_binary_hard.scatter_(1, top_k_indices, 1.0)
        
        # MODIFIED: Straight-Through Estimator
        # Forward pass: use hard (binary 0/1)
        # Backward pass: use soft (differentiable)
        allocation_binary = allocation_binary_hard - soft_allocation.detach() + soft_allocation
        
        # Map back to full allocation
        for batch_idx in range(batch_size):
            for sel_idx, full_idx in enumerate(selectable_indices):
                modality = full_idx // 12
                layer = full_idx % 12
                allocation_full[batch_idx, modality, layer] = allocation_binary[batch_idx, sel_idx]
        
        return allocation_full

class AdaptiveGestureClassifier(nn.Module):
    """
    Stage 2: Gesture Classifier with Adaptive Controller
    
    Architecture:
    1. QoI Perception: Extract quality features from inputs
    2. Layer Allocation: Decide which layers to activate
    3. Adaptive Backbones: Execute with selected layers
    4. Fusion & Classification: Same as Stage 1
    """
    
    def __init__(self, num_classes=4, adapter_hidden_dim=256, 
                 total_layers=8, qoi_dim=128, stage1_checkpoint=None):
        super(AdaptiveGestureClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.total_layers = total_layers
        
        # Fusion parameters (same as Stage 1)
        dim_dec = 256
        depth_dec = 6
        heads = 4
        
        # ========================================
        # QoI Perception Module (Trainable)
        # ========================================
        self.qoi_perception = QoIPerceptionModule(output_dim=qoi_dim)
        
        # ========================================
        # Layer Allocation Module (Trainable)
        # ========================================
        self.layer_allocator = LayerAllocationModule(
            qoi_dim=qoi_dim,
            hidden_dim=256,
            total_layers=total_layers
        )
        
        # ========================================
        # Backbones (From Stage 1)
        # ========================================
        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, 
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=0.0
        )
        
        self.depth = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=nn.LayerNorm, layerdrop=0.0
        )
        
        # ========================================
        # Adapters, Fusion, Classifier (From Stage 1)
        # ========================================
        self.vision_adapter = Adapter(768, adapter_hidden_dim)
        self.depth_adapter = Adapter(768, adapter_hidden_dim)
        
        self.encoder = TransformerEnc(
            dim=dim_dec, depth=depth_dec, heads=heads,
            dim_head=dim_dec//heads, mlp_dim=3*dim_dec
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(dim_dec, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Load Stage 1 weights if provided
        if stage1_checkpoint:
            self._load_stage1_weights(stage1_checkpoint)
    
    def _load_stage1_weights(self, checkpoint_path):
        """Load Stage 1 weights and freeze backbone/fusion/classifier"""
        print(f"Loading Stage 1 weights from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load weights (will ignore controller-related weights)
        msg = self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded Stage 1 weights: {msg}")
        
        # Freeze early layers, unfreeze last 4 layers
        print("\nFreezing strategy:")
        
        # RGB backbone: freeze first 8 layers, unfreeze last 4
        for i, block in enumerate(self.vision.blocks):
            if i < 8:  # Freeze layers 0-7
                for param in block.parameters():
                    param.requires_grad = False
            else:  # Unfreeze layers 8-11
                for param in block.parameters():
                    param.requires_grad = True
        
        # Depth backbone: same strategy
        for i, block in enumerate(self.depth.blocks):
            if i < 8:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Freeze patch embeddings
        for param in self.vision.patch_embed.parameters():
            param.requires_grad = False
        for param in self.depth.patch_embed.parameters():
            param.requires_grad = False
        
        # Unfreeze adapters
        for param in self.vision_adapter.parameters():
            param.requires_grad = True
        for param in self.depth_adapter.parameters():
            param.requires_grad = True
        
        # Unfreeze fusion encoder
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # Unfreeze classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"✅ Partial fine-tuning enabled")
        print(f"  Frozen params: {frozen_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
    
    def forward(self, rgb, depth, temperature=1.0, return_allocation=False):
        """
        Args:
            rgb: [batch, 3, 224, 224]
            depth: [batch, 3, 224, 224]
            temperature: Gumbel-Softmax temperature
            return_allocation: Whether to return layer allocation
        
        Returns:
            logits: [batch, num_classes]
            (optional) layer_allocation: [batch, 2, 12]
        """
        batch_size = rgb.size(0)
        
        # ========================================
        # Step 1: Perceive QoI
        # ========================================
        qoi_features = self.qoi_perception(rgb, depth)  # [batch, qoi_dim * 2]
        
        # ========================================
        # Step 2: Allocate Layers
        # ========================================
        layer_allocation = self.layer_allocator(qoi_features, temperature)  # [batch, 2, 12]
        
        # ========================================
        # Step 3: Execute Backbones with Selected Layers
        # ========================================
        outlist = []
        
        # RGB backbone
        dropped_layers_rgb = layer_allocation[:, 0, :]  # [batch, 12]
        rgb_features = self.vision.forward_controller(rgb, dropped_layers_rgb)
        rgb_features = torch.squeeze(rgb_features)
        if len(rgb_features.shape) == 1:
            rgb_features = torch.unsqueeze(rgb_features, dim=0)
        outlist.append(self.vision_adapter(rgb_features))
        
        # Depth backbone
        dropped_layers_depth = layer_allocation[:, 1, :]  # [batch, 12]
        depth_features = self.depth.forward_controller(depth, dropped_layers_depth)
        depth_features = torch.squeeze(depth_features)
        if len(depth_features.shape) == 1:
            depth_features = torch.unsqueeze(depth_features, dim=0)
        outlist.append(self.depth_adapter(depth_features))
        
        # ========================================
        # Step 4: Fusion & Classification (Same as Stage 1)
        # ========================================
        agg_features = torch.stack(outlist, dim=1)  # [batch, 2, 256]
        
        b, n, d = agg_features.shape
        agg_features = agg_features + positionalencoding1d(d, n)
        
        fused = self.encoder(agg_features)
        fused = torch.mean(fused, dim=1)  # [batch, 256]
        
        logits = self.classifier(fused)  # [batch, num_classes]
        
        if return_allocation:
            return logits, layer_allocation
        else:
            return logits


# Test the controller
if __name__ == "__main__":
    print("Testing AdaptiveGestureClassifier...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create model
    model = AdaptiveGestureClassifier(
        num_classes=4,
        adapter_hidden_dim=256,
        total_layers=8,
        qoi_dim=128,
        stage1_checkpoint='checkpoints/stage1/best_model.pth'
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    depth = torch.randn(batch_size, 3, 224, 224).to(device)
    
    model.eval()
    with torch.no_grad():
        logits, allocation = model(rgb, depth, temperature=1.0, return_allocation=True)
    
    print(f"\nForward pass test:")
    print(f"  Input RGB: {rgb.shape}")
    print(f"  Input Depth: {depth.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Layer allocation: {allocation.shape}")
    print(f"  Predictions: {torch.argmax(logits, dim=1)}")
    
    # Check allocation
    print(f"\nLayer allocation (first sample):")
    print(f"  RGB layers: {allocation[0, 0].cpu().numpy()}")
    print(f"  Depth layers: {allocation[0, 1].cpu().numpy()}")
    print(f"  Total active RGB: {allocation[0, 0].sum().item():.0f}")
    print(f"  Total active Depth: {allocation[0, 1].sum().item():.0f}")
    print(f"  Total active layers: {allocation[0].sum().item():.0f}/{model.total_layers}")
    
    print("\n✅ Controller test passed!")