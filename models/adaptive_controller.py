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
            rgb_qoi: [batch, output_dim]
            depth_qoi: [batch, output_dim]
        """
        # Extract QoI features
        rgb_feat = self.rgb_conv(rgb)  # [batch, 128, 1, 1]
        rgb_feat = rgb_feat.squeeze(-1).squeeze(-1)  # [batch, 128]
        rgb_qoi = self.rgb_proj(rgb_feat)  # [batch, output_dim]
        
        depth_feat = self.depth_conv(depth)
        depth_feat = depth_feat.squeeze(-1).squeeze(-1)
        depth_qoi = self.depth_proj(depth_feat)
        
        return rgb_qoi, depth_qoi


class LayerAllocationModule(nn.Module):
    """
    Allocate layers based on QoI features
    Uses Gumbel-Softmax for differentiable discrete sampling
    """
    def __init__(self, qoi_dim=128, total_layers=8):
        super(LayerAllocationModule, self).__init__()
        
        self.total_layers = total_layers
        
        # Fusion transformer to combine QoI info
        self.fusion = TransformerEnc(
            dim=qoi_dim, 
            depth=2,  # Lightweight fusion
            heads=4, 
            dim_head=qoi_dim//4, 
            mlp_dim=qoi_dim*2
        )
        
        # Output layer allocation logits
        # 24 = 12 (RGB layers) + 12 (Depth layers)
        self.layer_mlp = nn.Linear(qoi_dim, 24)
    
    # def forward(self, rgb_qoi, depth_qoi, temperature=1.0):
    #     """
    #     Args:
    #         rgb_qoi: [batch, qoi_dim]
    #         depth_qoi: [batch, qoi_dim]
    #         temperature: Gumbel-Softmax temperature
        
    #     Returns:
    #         layer_allocation: [batch, 2, 12] - binary allocation (0 or 1)
    #         raw_logits: [batch, 24] - for loss calculation
    #     """
    #     batch_size = rgb_qoi.size(0)
        
    #     # Stack QoI features: [batch, 2, qoi_dim]
    #     qoi_features = torch.stack([rgb_qoi, depth_qoi], dim=1)
        
    #     # Add positional encoding
    #     _, n, d = qoi_features.shape
    #     qoi_features = qoi_features + positionalencoding1d(d, n)
        
    #     # Fuse QoI information
    #     fused = self.fusion(qoi_features)  # [batch, 2, qoi_dim]
    #     fused = torch.mean(fused, dim=1)  # [batch, qoi_dim]
        
    #     # Get layer allocation logits
    #     raw_logits = self.layer_mlp(fused)  # [batch, 24]
        
    #     # Gumbel-Softmax sampling with top-L selection
    #     # Apply Gumbel-Softmax to get soft allocation
    #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(raw_logits) + 1e-10) + 1e-10)
    #     logits_with_noise = (raw_logits + gumbel_noise) / temperature
    #     soft_allocation = F.softmax(logits_with_noise, dim=-1)
        
    #     # Top-L selection (select top `total_layers` from 24 layers)
    #     _, top_indices = torch.topk(soft_allocation, self.total_layers, dim=-1)
        
    #     # Create hard allocation (straight-through estimator)
    #     hard_allocation = torch.zeros_like(soft_allocation)
    #     hard_allocation.scatter_(1, top_indices, 1.0)
        
    #     # Straight-through: forward uses hard, backward uses soft
    #     allocation = hard_allocation - soft_allocation.detach() + soft_allocation
        
    #     # Reshape to [batch, 2, 12] (RGB: first 12, Depth: last 12)
    #     layer_allocation = torch.zeros(batch_size, 2, 12).to(allocation.device)
    #     layer_allocation[:, 0, :] = allocation[:, :12]   # RGB layers
    #     layer_allocation[:, 1, :] = allocation[:, 12:]   # Depth layers
        
    #     return layer_allocation, raw_logits

    def forward(self, rgb_qoi, depth_qoi, temperature=1.0):
        """
        Args:
            rgb_qoi: [batch, qoi_dim]
            depth_qoi: [batch, qoi_dim]
            temperature: Gumbel-Softmax temperature
        
        Returns:
            layer_allocation: [batch, 2, 12] - binary allocation (0 or 1)
            raw_logits: [batch, 24] - for loss calculation
        """
        batch_size = rgb_qoi.size(0)
        device = rgb_qoi.device  # ⭐ Get device from input
        
        # Stack QoI features: [batch, 2, qoi_dim]
        qoi_features = torch.stack([rgb_qoi, depth_qoi], dim=1)
        
        # Add positional encoding
        _, n, d = qoi_features.shape
        qoi_features = qoi_features + positionalencoding1d(d, n)
        
        # Fuse QoI information
        fused = self.fusion(qoi_features)  # [batch, 2, qoi_dim]
        fused = torch.mean(fused, dim=1)  # [batch, qoi_dim]
        
        # Get layer allocation logits
        raw_logits = self.layer_mlp(fused)  # [batch, 24]
        
        # ⭐ CRITICAL: Always activate first layer of each backbone
        # Reserve 2 layers (1 RGB + 1 Depth) for first layers
        remaining_budget = self.total_layers - 2
        
        # Create mask for selectable layers (all except first layers)
        # RGB layers: 0-11, Depth layers: 12-23
        # First layers: 0 (RGB), 12 (Depth)
        selectable_indices = []
        for i in range(24):
            if i != 0 and i != 12:  # Skip first layers
                selectable_indices.append(i)
        
        selectable_indices = torch.tensor(selectable_indices, device=device)  # [22]
        
        # Extract logits for selectable layers
        selectable_logits = raw_logits[:, selectable_indices]  # [batch, 22]
        
        # Gumbel-Softmax sampling on selectable layers
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(selectable_logits) + 1e-10) + 1e-10)
        logits_with_noise = (selectable_logits + gumbel_noise) / temperature
        soft_allocation_selectable = F.softmax(logits_with_noise, dim=-1)
        
        # Top-(L-2) selection from 22 selectable layers
        _, top_indices = torch.topk(soft_allocation_selectable, remaining_budget, dim=-1)
        
        # Create hard allocation for selectable layers
        hard_allocation_selectable = torch.zeros_like(soft_allocation_selectable)
        hard_allocation_selectable.scatter_(1, top_indices, 1.0)
        
        # Straight-through estimator for selectable layers
        allocation_selectable = (hard_allocation_selectable - 
                            soft_allocation_selectable.detach() + 
                            soft_allocation_selectable)
        
        # Reconstruct full allocation [batch, 24]
        allocation_full = torch.zeros(batch_size, 24, device=device)
        allocation_full[:, 0] = 1.0   # RGB first layer always 1
        allocation_full[:, 12] = 1.0  # Depth first layer always 1
        
        # Fill in selected layers
        for i, idx in enumerate(selectable_indices):
            allocation_full[:, idx] = allocation_selectable[:, i]
        
        # Reshape to [batch, 2, 12] (RGB: first 12, Depth: last 12)
        layer_allocation = torch.zeros(batch_size, 2, 12, device=device)
        layer_allocation[:, 0, :] = allocation_full[:, :12]   # RGB layers
        layer_allocation[:, 1, :] = allocation_full[:, 12:]   # Depth layers
        
        return layer_allocation, raw_logits
    
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
            total_layers=total_layers
        )
        
        # ========================================
        # Backbones (Frozen from Stage 1)
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
        # Adapters, Fusion, Classifier (Frozen from Stage 1)
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
        
        # Load and freeze Stage 1 weights
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
        
        # Unfreeze adapters (they need to adapt)
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
        rgb_qoi, depth_qoi = self.qoi_perception(rgb, depth)
        
        # ========================================
        # Step 2: Allocate Layers
        # ========================================
        layer_allocation, raw_logits = self.layer_allocator(
            rgb_qoi, depth_qoi, temperature
        )  # [batch, 2, 12]
        
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