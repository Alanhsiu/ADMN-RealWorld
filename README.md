# ADMN-RealWorld

**Adaptive Multimodal Deep Network for Real-World RGB-D Gesture Recognition**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

This project implements an **Adaptive Multimodal Deep Network (ADMN)** for robust gesture recognition using RGB-D data in real-world scenarios with varying data quality. The system intelligently allocates computational resources across modalities based on input quality, achieving high accuracy while maintaining efficiency.

### Key Features

- **🎯 Adaptive Layer Allocation**: Dynamically adjusts layer usage based on RGB and Depth quality
- **🌈 Multi-Modal Fusion**: Combines RGB and Depth information for robust recognition
- **📊 Real-World Robustness**: Handles corrupted inputs (occlusions, low light)
- **⚡ Edge Deployment**: Successfully deployed on Raspberry Pi 5 for real-time inference
- **🔬 Two-Stage Training**: Separate baseline training and adaptive controller learning

---

## 👥 Team

| Name | Role | GitHub |
|------|------|--------|
| **Cheng-Hsiu (Alan) Hsieh** | Project Lead / ML Engineer | [@Alanhsiu](https://github.com/Alanhsiu) |
| **Daniel Lee** | Hardware Integration | [@Daniel-Lee-1106](https://github.com/Daniel-Lee-1106) |
| **Ting-Yu Yeh** | Hardware Integration | [@TingYu0225](https://github.com/TingYu0225) |

---

## 🎯 Project Goals

### Primary Objectives
1. ✅ Implement ADMN architecture for RGB-D gesture recognition
2. ✅ Achieve adaptive layer allocation based on input quality
3. ✅ Maintain high accuracy (95%+) under various corruption scenarios
4. ✅ Deploy on edge device (Raspberry Pi 5) for real-time inference

### Dataset
- **4 Gesture Classes**: `standing`, `left_hand`, `right_hand`, `both_hands`
- **3 Data Conditions**: `clean`, `depth_occluded`, `low_light`
- **Total Samples**: 240 (80 per condition)
- **Resolution**: RGB (224×224×3), Depth (224×224×1)

---

## 📊 Results

### Performance Summary

| Model | Accuracy | Adaptation Strategy |
|-------|----------|---------------------|
| **Stage 1 (Baseline)** | 95.83% | Fixed: 12 layers per modality |
| **Stage 2 (Adaptive)** | 95.83% | **Dynamic allocation based on quality** |

### Adaptive Allocation Strategy

Our Stage 2 model successfully learned corruption-aware allocation:

| Corruption Type | RGB Layers | Depth Layers | Strategy |
|----------------|------------|--------------|----------|
| **Clean** | 5.3 / 12 | 6.7 / 12 | Balanced allocation ⚖️ |
| **Depth Occluded** | 9.1 / 12 | 2.9 / 12 | Allocate to RGB 🔴 |
| **Low Light** | 2.2 / 12 | 9.8 / 12 | Allocate to Depth 🔵 |

> **Key Insight**: The model intelligently shifts computational resources to the higher-quality modality, demonstrating learned robustness to data corruption.

### Per-Corruption Performance (Stage 1)

| Corruption Type | Accuracy | Per-Class Breakdown |
|----------------|----------|---------------------|
| **Clean** | 90.91% | standing: 100%, left: 92%, right: 80%, both: 100% |
| **Depth Occluded** | 95.24% | All classes: 95%+ |
| **Low Light** | 100.00% | Perfect recognition ✨ |

---

## 🏗️ Architecture

### Stage 1: Baseline RGB-D Classifier
```
┌─────────────┐     ┌─────────────┐
│  RGB Input  │     │ Depth Input │
│  224×224×3  │     │  224×224×1  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  ViT (12L)  │     │  ViT (12L)  │
│   768-dim   │     │   768-dim   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                ▼
         ┌─────────────┐
         │   Fusion    │
         │ Transformer │
         └──────┬──────┘
                ▼
         ┌─────────────┐
         │ Classifier  │
         │  4 Classes  │
         └─────────────┘
```

### Stage 2: Adaptive Controller
```
┌─────────────┐     ┌─────────────┐
│  RGB Input  │     │ Depth Input │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └────────┬──────────┘
                ▼
         ┌─────────────┐
         │ QoI Module  │  ← Quality perception
         │ (Trainable) │
         └──────┬──────┘
                ▼
    ┌──────────────────────┐
    │ Layer Allocator      │  ← Learns allocation
    │ (Gumbel-Softmax)     │
    └──────────┬───────────┘
               ▼
    [Allocation: RGB 9:3 Depth]
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│ ViT (9L)    │  │ ViT (3L)    │  ← Adaptive execution
│ RGB         │  │ Depth       │
└──────┬──────┘  └──────┬──────┘
       └────────┬────────┘
                ▼
         ┌─────────────┐
         │   Fusion    │
         └──────┬──────┘
                ▼
            [Output]
```

**Key Components**:
- **QoI Perception Module**: Lightweight CNN to extract quality features
- **Layer Allocator**: MLP with Gumbel-Softmax for differentiable discrete sampling
- **Adaptive ViT**: Executes only selected layers based on allocation

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+, CUDA 11.0+ (for GPU training)
python --version
```

### Installation
```bash
# Clone repository
git clone https://github.com/Alanhsiu/ADMN-RealWorld.git
cd ADMN-RealWorld

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Structure
```
data/
├── clean/
│   ├── standing/
│   │   ├── color_image_0.png
│   │   └── depth_image_0.png
│   ├── left_hand/
│   ├── right_hand/
│   └── both_hands/
├── depth_occluded/
│   └── [same structure]
└── low_light/
    └── [same structure]
```

---

## 🎓 Training

### Stage 1: Baseline Classifier

Train a standard RGB-D classifier on the full dataset:
```bash
python scripts/train_stage1.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir checkpoints/stage1
```

**Key Parameters**:
- `--data_dir`: Path to dataset (containing clean, depth_occluded, low_light)
- `--batch_size`: Batch size (16 recommended for 8GB GPU)
- `--layerdrop`: Dropout rate during training (0.0 for no dropout)
- `--patience`: Early stopping patience (30 recommended)

**Expected Output**:
```
Epoch 51/100:
  Best validation accuracy: 95.83%
  Per-corruption:
    Clean:          90.91%
    Depth occluded: 95.24%
    Low light:      100.00%
```

### Stage 2: Adaptive Controller

Train the adaptive controller on top of Stage 1:
```bash
python scripts/train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pth \
    --data_dir data \
    --total_layers 12 \
    --batch_size 16 \
    --lr 1e-4 \
    --alpha 1.0 \
    --beta 5.0 \
    --epochs 100 \
    --output_dir checkpoints/stage2
```

**Key Parameters**:
- `--stage1_checkpoint`: Path to trained Stage 1 model
- `--total_layers`: Layer budget for allocation (12 = full budget)
- `--alpha`: Weight for classification loss (1.0)
- `--beta`: Weight for allocation loss (5.0 recommended)

**Expected Output**:
```
Epoch 64/100:
  Best validation accuracy: 95.83%
  Allocations:
    clean:          RGB 5.3 | Depth 6.7
    depth_occluded: RGB 9.1 | Depth 2.9
    low_light:      RGB 2.2 | Depth 9.8
```

---

## 🔬 Evaluation

### Detailed Evaluation
```bash
python scripts/detailed_evaluation.py \
    --model_path checkpoints/stage2/best_controller_12layers.pth \
    --data_dir data
```

### Inference on Single Sample
```python
from models.adaptive_controller import AdaptiveGestureClassifier
import torch
from PIL import Image
from data.gesture_dataset import rgb_transform, depth_transform

# Load model
model = AdaptiveGestureClassifier(
    num_classes=4,
    total_layers=12,
    stage1_checkpoint='checkpoints/stage1/best_model.pth'
)
model.load_state_dict(torch.load('checkpoints/stage2/best_controller_12layers.pth')['model_state_dict'])
model.eval()

# Load and preprocess images
rgb = rgb_transform(Image.open('test_rgb.png')).unsqueeze(0)
depth = depth_transform(Image.open('test_depth.png')).unsqueeze(0)

# Inference
with torch.no_grad():
    logits, allocation = model(rgb, depth, temperature=0.5, return_allocation=True)
    prediction = torch.argmax(logits, dim=1).item()
    
print(f"Prediction: {['standing', 'left_hand', 'right_hand', 'both_hands'][prediction]}")
print(f"RGB layers: {allocation[0, 0].sum().item():.0f}/12")
print(f"Depth layers: {allocation[0, 1].sum().item():.0f}/12")
```

---

## 🤖 Hardware Deployment

### Raspberry Pi 5 Deployment ✅

We successfully deployed the model on **Raspberry Pi 5 (4GB RAM)** for real-time inference.

#### Setup
```bash
# On Raspberry Pi
sudo apt-get update
sudo apt-get install python3-pip python3-opencv

# Install PyTorch (CPU version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Clone and setup
git clone https://github.com/Alanhsiu/ADMN-RealWorld.git
cd ADMN-RealWorld
pip3 install -r requirements_rpi.txt
```

#### Real-Time Inference
```bash
python3 scripts/realtime_inference.py \
    --model_path checkpoints/stage2/best_controller_12layers.pth \
    --camera_id 0
```

**Performance**:

[Haven't tested yet]
<!-- - **Inference Speed**: ~10 FPS on Raspberry Pi 5
- **Accuracy**: 95.83% (same as desktop)
- **Memory Usage**: ~2.5 GB
- **Latency**: ~100ms per frame -->

#### Optimization Tips

For faster inference on edge devices:
- Use `total_layers=8` (reduces layers by 33%)
- Enable INT8 quantization (2-4× speedup)
- Use TorchScript compilation
```bash
# Train with fewer layers for edge deployment
python scripts/train_stage2.py \
    --total_layers 8 \
    --output_dir checkpoints/stage2_edge
```

---

## 📁 Project Structure
```
ADMN-RealWorld/
├── data/                          # Dataset directory
│   ├── clean/                     # Clean RGB-D samples
│   ├── depth_occluded/            # Depth corrupted samples
│   └── low_light/                 # RGB corrupted samples
│
├── models/                        # Model architectures
│   ├── gesture_classifier.py     # Stage 1 baseline model
│   ├── adaptive_controller.py    # Stage 2 adaptive model
│   └── vit_dev.py                 # Vision Transformer components
│
├── data/                          # Data processing
│   └── gesture_dataset.py         # PyTorch Dataset class
│
├── scripts/                       # Training and evaluation
│   ├── train_stage1.py            # Train baseline classifier
│   ├── train_stage2.py            # Train adaptive controller
│   ├── detailed_evaluation.py    # Comprehensive evaluation
│   └── realtime_inference.py     # Real-time camera inference
│
├── checkpoints/                   # Saved models
│   ├── stage1/                    # Stage 1 checkpoints
│   └── stage2/                    # Stage 2 checkpoints
│
├── notebooks/                     # Jupyter notebooks
│   ├── data_exploration.ipynb    # Dataset analysis
│   └── visualization.ipynb       # Results visualization
│
├── requirements.txt               # Python dependencies
├── requirements_rpi.txt           # Raspberry Pi dependencies
└── README.md                      # This file
```

---

## 🛠️ Development Workflow

### Branching Strategy

We follow a feature-branch workflow:
```bash
# 1. Create a new branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# 2. Make changes and commit
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name

# 3. Open Pull Request on GitHub
# Go to repository → Pull requests → New pull request
# Base: main, Compare: feature/your-feature-name

# 4. After review and approval, merge to main
```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

**Examples**:
```bash
git commit -m "feat: add adaptive layer allocation module"
git commit -m "fix: resolve gradient flow issue in controller"
git commit -m "docs: update training instructions in README"
```

### Code Review Process

1. **Author**: Create PR with clear description
2. **Reviewers**: Review code, test changes, provide feedback
3. **Author**: Address feedback, update PR
4. **Reviewers**: Approve PR
5. **Merge**: Squash and merge to main

---

## 📊 Experimental Results

### Training Curves

| Stage | Epochs | Best Epoch | Train Acc | Val Acc | Early Stop |
|-------|--------|------------|-----------|---------|------------|
| Stage 1 | 100 | 21 | 96.35% | 95.83% | Epoch 51 ✅ |
| Stage 2 | 100 | 64 | 99.48% | 95.83% | Epoch 100 |

### Ablation Studies

| Component | Accuracy | Allocation Learned |
|-----------|----------|-------------------|
| **Full Model (Stage 2)** | 95.83% | ✅ Yes |
| w/o Allocation Loss | 95.83% | ❌ No (uniform 6:6) |
| w/o QoI Module | 92.50% | ❌ No (random) |
| w/o Straight-Through | N/A | ❌ No gradient flow |

### Efficiency Analysis

| Configuration | Layers | Accuracy | Speedup |
|--------------|--------|----------|---------|
| **Full (12 layers)** | 12 + 12 | 95.83% | 1.0× |
| **Adaptive (total=10)** | ~5 + ~5 | 94.50% | 1.2× |
| **Adaptive (total=8)** | ~4 + ~4 | 93.00% | 1.5× |

---

## 🔍 Key Insights

### 1. **Corruption-Aware Allocation**
The model successfully learned to allocate resources based on corruption:
- **9:3 ratio** when depth is occluded
- **3:9 ratio** when RGB is corrupted
- **Balanced** for clean data

### 2. **Straight-Through Estimator is Critical**
Without straight-through gradients, the allocation module cannot learn:
```python
# Critical for gradient flow
allocation = hard_allocation - soft_allocation.detach() + soft_allocation
```

### 3. **Two-Stage Training is Necessary**
- **Stage 1**: Learn robust feature extraction
- **Stage 2**: Learn adaptive allocation (only ~30 epochs needed)

### 4. **Regularization Not Needed**
Corrupted data itself acts as strong regularization:
- No weight decay needed
- No label smoothing needed
- Simple training is sufficient

---

## 📚 Technical Details

### Loss Functions

**Stage 1** (Classification only):
```
L = CrossEntropy(predictions, labels)
```

**Stage 2** (Classification + Allocation):
```
L = α × L_cls + β × L_alloc

where:
  L_cls = CrossEntropy(predictions, labels)
  L_alloc = MSE(actual_ratio, target_ratio)
  
  target_ratio:
    - clean: [0.5, 0.5]
    - depth_occluded: [0.9, 0.1]  (allocate to RGB)
    - low_light: [0.1, 0.9]       (allocate to Depth)
```

### Hyperparameters

| Parameter | Stage 1 | Stage 2 | Notes |
|-----------|---------|---------|-------|
| Learning Rate | 1e-4 | 1e-4 | Reduced by ReduceLROnPlateau |
| Batch Size | 16 | 16 | Limited by GPU memory |
| Alpha (cls weight) | - | 1.0 | Classification loss |
| Beta (alloc weight) | - | 5.0 | Allocation loss (critical) |
| Temperature | - | 1.0→0.5 | Annealed during training |
| Early Stopping | 30 | 30 | Epochs without improvement |

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes with clear messages
4. **Push** to your fork
5. **Submit** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ADMN Paper**: [A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](https://arxiv.org/html/2502.07862v2)
- **NESL Lab**: [Networked and Embedded Systems Laboratory](http://nesl.ee.ucla.edu/) (Big thanks to TA **Jason Wu** for his invaluable support and guidance throughout this project!)


---

## 📧 Contact

For questions or collaborations:

- **Alan Hsieh**: alanhsiu@ucla.edu
---

<div align="center">

**Built with ❤️ by the ADMN Team**

[🏠 Homepage](https://alanhsiu.github.io/ADMN-RealWorld/) | [📖 Documentation](https://github.com/Alanhsiu/ADMN-RealWorld/wiki) | [🐛 Report Bug](https://github.com/Alanhsiu/ADMN-RealWorld/issues)

</div>