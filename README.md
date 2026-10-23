# ELEC 475: Lab 2 Starter Template
## From LeNet-5 to AlexNet: An Architectural & Empirical Evolution

Welcome to ELEC 475 Lab 2! This repository contains the starter code and scaffolding for both your **In-Lab (2-hour milestone)** and **Post-Lab (due Oct 22)** assignments.

### Directory Structure

```text
lab2_template/
├── configs/
│   └── config.py          # Centralized configuration dataclass (ExperimentConfig)
├── models/
│   ├── lenet.py           # TODO [In-Lab Step 2]: Base LeNet-5 implementation
│   ├── lenet_modern.py    # TODO [In-Lab Step 4]: Modernized LeNet (ReLU + Overlapping MaxPool)
│   ├── alexnet.py         # TODO [Post-Lab Step 1]: Full 5-stage AlexNet with LRN/BatchNorm
│   └── zoo.py             # Pre-built PyTorch Model Zoo adapter (torchvision.models.alexnet)
├── utils/
│   ├── dataset.py         # Automated CIFAR-10 and Tiny ImageNet loaders
│   ├── eda.py             # Exploratory Data Analysis tool (class distributions, RGB stats, imbalance ratio)
│   ├── transforms.py      # TODO [Post-Lab Step 3]: PyTorch GPU data augmentation pipeline
│   ├── metrics.py         # Loss, Top-1 accuracy, Macro/Micro F1, ConfusionMatrixTracker & EarlyStopping
│   ├── seed.py            # Global deterministic seeding across Python, NumPy, PyTorch (CPU/CUDA)
│   ├── visualization.py   # Training curve and ablation plotters
│   └── profiler.py        # Timer (CUDA sync), ModelProfiler (FLOPs/MACs/Params)
├── compare_models.py      # Multi-model complexity & FLOPs benchmarking CLI script
├── demo_transforms.py     # Pre-Lab benchmark: PyTorch GPU Tensors vs OpenCV vs NumPy
├── train.py               # Complete CLI training engine with Timer, ModelProfiler, Macro F1 & Early Stopping
├── evaluate.py            # Checkpoint evaluation script with full classification report
└── requirements.txt       # Python package dependencies (torch, torchvision, thop, torchinfo)
```

---

### Step-by-Step Execution Guide

#### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell (or source .venv/bin/activate on Linux/WSL/macOS)
pip install -r requirements.txt
```

#### 2. Run Pre-Lab Exploratory Data Analysis & Verification
```powershell
# 1. Run Comprehensive Dual-Dataset Exploratory Data Analysis (EDA):
python utils/eda.py --dataset both

# 2. Run Automated Environment & Dataset Verification:
python prelab_verify.py
```

#### 3. In-Lab Milestone (2 Hours)
- Open `models/lenet.py` and implement the `TODO [IN-LAB STEP 2]` sections.
- Verify architecture: `python -m models.lenet`
- Train Base LeNet (real-time epoch timers and FLOPs logged):
  ```powershell
  python train.py --model lenet --dataset cifar10 --epochs 5 --save-model checkpoints/lenet_baseline.pth --plot-file plots/loss_lenet.png
  ```
- Open `models/lenet_modern.py` and implement the `TODO [IN-LAB STEP 4]` sections.
- Verify architecture: `python -m models.lenet_modern`
- Train Modernized LeNet:
  ```powershell
  python train.py --model lenet_modern --dataset cifar10 --epochs 5 --save-model checkpoints/lenet_modern.pth --plot-file plots/loss_lenet_modern.png
  ```
- Demonstrate results to a TA for **In-Lab Checkoff (3 Marks)**!

#### 4. Post-Lab (Due Oct 22, 23:59 ET)
Everything is automated (GPU recommended, see `Lab2_Colab.ipynb`):
```
pip install -r requirements.txt
python run_all.py                  # all CIFAR-10 / Tiny ImageNet runs, seeds, extended ablation, tools
python make_report.py --name "Your Name" --student-number 12345678
python package_submission.py --student-number 12345678
```
`commands.txt` lists every command that `run_all.py` executes (exact `train.py` calls). Use `python run_all.py --smoke` for a 2-epoch dry run.
