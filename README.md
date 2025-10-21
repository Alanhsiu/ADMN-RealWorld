# ADMN-RealWorld

Adaptive Multimodal Network for Real-World Gesture Recognition

## Team
- Cheng-Hsiu (Alan) Hsieh
- Daniel Lee
- Ting-Yu Yeh

## Project Goal
Implement ADMN on real RGB-D gesture data:
- 4 gesture classes: standing, left hand, right hand, both hands
- Adaptive layer allocation based on input quality
- Target: reduce FLOPs while maintaining accuracy

## Status: 🚧 In Progress

### Current Phase: Data Preparation
- [ ] Organize collected RGB-D data
- [ ] Split train/val/test sets
- [ ] Create PyTorch Dataset class

---

## Quick Setup
```bash
git clone https://github.com/Alanhsiu/ADMN-RealWorld.git
cd ADMN-RealWorld
git checkout [your-branch-name]  # Switch to development branch
```

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
which python
pip install --upgrade pip
pip install numpy scikit-learn opencv-python Pillow
```

## Contribution Guidelines

#### 1. Create a new branch
```bash
git checkout main
git pull origin main
git checkout -b [your-branch-name]
```

#### 2. Make changes and commit
```bash
git add .
git commit -m "[your-commit-message]"
git push origin [your-branch-name]
```

#### 3. Open PR on GitHub
- Go to repo → "Pull requests" → "New pull request"
- Base: main, Compare: feature/your-task
- Add title and description
- Click "Create pull request"

#### 4. Wait for Others' review
- Wait for others to review your changes
- If approved, merge your changes into main
- If not approved, make changes and commit again

#### 5. For Reviewers
- Review PR on GitHub, and merge the changes into main
```bash
git checkout main
git pull origin main
git checkout [your-branch-name]
git merge main
git push origin [your-branch-name]
```