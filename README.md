# bh-VLA: Unified Vision-Language-Action Model Training Framework

<div align="center">

**Two state-of-the-art VLA policies in one unified framework**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

## 📋 Table of Contents

- [What is this?](#what-is-this)
- [Paper References](#paper-references)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Collect Data](#1-collect-data)
  - [2. Train](#2-train)
  - [3. Run Inference](#3-run-inference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## What is this?

**bh-VLA** is a unified framework that implements **two Vision-Language-Action (VLA) policies** for robot control, both of which can be trained and run on the **SO-101** (ALOHA-compatible) robot arm.

### Two Policies, One Codebase

| Policy | Paper | Architecture | Best For |
|--------|-------|--------------|----------|
| **ACT** | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | Transformer decoder, predicts action chunks | Quick training, specific tasks |
| **pi0.5** | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) | PaliGemma 3B backbone + flow matching | Open-world generalization |

### Why Two Policies?

- **ACT** is faster to train and simpler. Good for learning specific tasks (e.g., "pick up the cup").
- **pi0.5** is more powerful. It can generalize to new tasks and environments, but requires more compute.

You can switch between them with a single flag:

```bash
python train.py --mode act      # Use ACT
python train.py --mode pi05     # Use pi0.5
```

---

## Paper References

### 1. ACT (Action Chunking Transformers)
**Paper:** ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705)

- **Authors:** Tony Z. Zhao et al.
- **Key insight:** Predict a chunk of N future actions at once (instead of one at a time) to reduce error accumulation
- **Robot:** ALOHA (6-DOF bimanual arms)
- **Architecture:** ResNet image encoder → Transformer decoder → Action prediction head

### 2. pi0.5 (Physical Intelligence)
**Paper:** ["pi0.5: A Vision-Language-Action Model with Open-World Generalization"](https://arxiv.org/abs/2504.16054)

- **Authors:** Physical Intelligence team
- **Key insight:** Use a large Vision-Language Model (PaliGemma 3B) as backbone with flow matching for action prediction
- **Generalization:** Can clean kitchens/bedrooms in entirely new homes
- **Architecture:** PaliGemma VLM → Action expert → Flow matching

### 3. SO-101 Robot Hardware
**Reference:** [LeRobot SO-101 Documentation](https://huggingface.co/docs/lerobot/so101)

- **Robot:** SO-101 (by The Robot Studio)
- **Components:** 6-DOF leader arm + 6-DOF follower arm + 3 cameras + Feetech STS3215 servos
- **Teleoperation:** Human operator moves the leader arm, follower arm mirrors movements

---

## Quick Start

This is a complete VLA training framework. Follow these steps in order:

### Step 1: Install Dependencies

```bash
cd bh-VLA

# Install uv if not already installed
pip install uv

# Create virtual environment and install all dependencies
uv venv && source .venv/bin/activate   # macOS/Linux
uv venv && .venv\Scripts\activate      # Windows

uv pip install -e .                   # 본 패키지 + deps

# Optional extras (unneeded modules 제거 시 스킵)
uv pip install -e '.[robot]'           # 로봇 하드웨어 (pyserial, lerobot)
uv pip install -e '.[rlds]'            # RLDS 데이터셋 포맷
```

### Step 2: (Optional) Set Up Robot Hardware

If you have an SO-101 robot:

```bash
# Install LeRobot (for robot communication)
pip install lerobot

# Find USB ports
lerobot-find-port

# Set up motor IDs
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM1
```

### Step 3: Collect Training Data

```bash
python collect_data.py --mode act --num-episodes 5
```

### Step 4: Train Your Policy

```bash
# Train ACT (single GPU)
python train.py --mode act

# Train pi0.5 (single GPU)
python train.py --mode pi05

# Train pi0.5 — **multi-GPU recommended** ⚡ (pi0.5는 GPU 2개 이상 권장)
accelerate launch train.py --mode pi05 --use-accelerate

# Multi-GPU 4개 GPU + gradient accumulation 4
accelerate launch --num_processes=4 train.py --mode pi05 --use-accelerate --gradient-accumulation 4
```

**GPU 추천 사양:**
| Policy | 최소 GPU | 권장 GPU |
|--------|----------|----------|
| ACT | RTX 3060 12GB | RTX 3060 12GB |
| pi0.5 | RTX 4090 24GB | 2× RTX 4090 / A100 80GB |

*pi0.5는 큰 모델(PaliGemma 3B backbone)이라 멀티 GPU 없이는 VRAM이 부족할 수 있어요.*

### Step 5: Run Inference

```bash
# On real robot
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

# Test with synthetic data (no robot needed)
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode
```

---

## Architecture

### ACT Policy Architecture

```
                    ┌──────────────┐
                    │ Image Input  │
                    │ (3 cameras)  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ ResNet-18    │ ← Visual feature extractor
                    │ Encoder      │ (512-dim features per camera)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Language     │ ← Text instruction encoder
                    │ Encoder      │ (BERT-style token embedding)
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Concatenate  │ ← Combine image + text features
                    │ + Project    │ (to transformer hidden size)
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │ Transformer Decoder     │ ← Core of ACT
              │ (4 layers, 8 heads)     │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │ Action Head  │ ← Predict N future actions
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Action Chunk │ ← Output: (N timesteps × 14 dim)
                    └──────────────┘
```

### pi0.5 Architecture

```
                    ┌──────────────┐
                    │ Image Input  │
                    │ (3 cameras)  │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │ SigLIP ViT-L/14         │ ← Vision encoder (~307M params)
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │ Multimodal Projector    │ ← Align vision → text space
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │ PaliGemma   │ ← Large VLM backbone (3B params)
                    │ LLM decoder │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │ Action Expert MLP       │ ← Specialized for robot control
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │ Flow Matching           │ ← Generative action prediction
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │ Action       │ ← Output with uncertainty
                    └──────────────┘
```

### Key Differences

| Aspect | ACT | pi0.5 |
|--------|-----|-------|
| **Backbone** | ResNet-18 + small transformer | PaliGemma 3B VLM |
| **Action prediction** | Direct regression | Flow matching (generative) |
| **Training loss** | MSE | Flow matching loss |
| **Training time** | Fast (hours) | Slow (days) |
| **Generalization** | Task-specific | Open-world |
| **Compute** | Single GPU | Multi-GPU recommended |

---

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for pi0.5; ACT can run on CPU)
- CUDA 11.8+ (if using GPU)

### 1. Clone and Install

```bash
cd bh-VLA
pip install -r requirements.txt
```

### 2. (Optional) Install Robot Libraries

If you have SO-101 hardware:

```bash
pip install lerobot
pip install lerobot[feetech]
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print('OpenCV: OK')"
python -c "import numpy; print('NumPy: OK')"
```

---

## Usage

### 1. Collect Data

Collect demonstration data using the leader arm (teleoperation):

```bash
# Collect 10 episodes for ACT training
python collect_data.py --mode act --num-episodes 10 --task-name "pick_and_place"

# Collect data for pi0.5
python collect_data.py --mode pi05 --num-episodes 20 --task-name "fold_clothes"
```

**What happens during data collection:**

1. You specify the task (e.g., "pick up the red cup")
2. You move the leader arm to demonstrate the task
3. The follower arm mirrors your movements
4. Images from all 3 cameras, joint positions, and timestamps are recorded
5. Data is saved as numpy arrays in `data/{task_name}/episode_{NNNN}/`

**Data format per episode:**

```
data/{task_name}/episode_0000/
├── images.npy       # Shape: (frames, H, W, C)
├── states.npy       # Shape: (frames, 28) - joint positions
├── actions.npy      # Shape: (frames, 28) - same as states for teleop
└── metadata.json    # Task name, frame count, duration
```

### 2. Train

Train your policy with the unified training interface:

```bash
# ACT training (default)
python train.py --mode act

# pi0.5 training
python train.py --mode pi05

# Custom hyperparameters
python train.py --mode act --lr 1e-4 --batch-size 16 --epochs 50

# Use custom dataset
python train.py --mode act --data-dir ./data/pick_and_place

# Use GPU (default)
python train.py --mode act --device cuda

# Resume training from checkpoint
python train.py --mode act --resume
```

**Training output:**

```
============================================================
  bh-VLA Training Framework
  Mode: act
  Device: cuda
============================================================
Total parameters: 1,234,567
------------------------------------------------------------
Training ACT policy...
  Device: cuda:0
  Dataset size: 10 episodes
  Batch size: 32
  Learning rate: 1e-04
  Epochs: 100
------------------------------------------------------------
  Epoch [1/100] Batch [1/1] Loss: 0.123456 LR: 1.00e-04 Time: 2.5s
  Epoch [1/100] Batch [2/1] Loss: 0.098765 LR: 2.00e-04 Time: 2.4s
...
Epoch 10/100 complete. Average Loss: 0.005432
Checkpoint saved to ./checkpoints/act_epoch_10.pt
...
Training complete! Final checkpoint saved to ./checkpoints/
```

### 3. Run Inference

Run the trained policy on the robot:

```bash
# On real robot
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

# Test with synthetic data (no robot needed)
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode

# With custom language instruction
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt \
    --language "pick up the red cup"

# pi0.5 inference with flow matching
python inference.py --mode pi05 --checkpoint ./checkpoints/pi05_last.pt \
    --flow-steps 50
```

**What happens during inference:**

1. Policy loads the checkpoint
2. Robot connects (cameras + motors)
3. For each step (~50Hz):
   - Capture images from cameras
   - Run policy forward pass
   - Send action to follower arm motors
4. Action history is saved to `outputs/action_history.npy`

---

## Configuration

### ACT Hyperparameters (train.py line ~30)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_encoder` | resnet18 | Visual feature extractor |
| `image_size` | 224 | Input image resolution |
| `text_vocab_size` | 30522 | Text vocabulary size |
| `transformer_hidden_size` | 512 | Transformer hidden dimension |
| `transformer_num_layers` | 4 | Number of transformer layers |
| `transformer_num_heads` | 8 | Number of attention heads |
| `action_chunk_size` | 90 | Future timesteps to predict |
| `lr` | 1e-4 | Learning rate |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 100 | Total training epochs |

### pi0.5 Hyperparameters (train.py line ~100)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | paligemma3b | VLM backbone |
| `flow_steps` | 500 | Flow matching integration steps |
| `flow_sigma` | 0.02 | Gaussian noise level |
| `action_chunk_size` | 32 | Future timesteps to predict |
| `lr` | 1e-5 | Learning rate (smaller for fine-tuning) |
| `batch_size` | 8 | Training batch size |
| `num_epochs` | 50 | Total training epochs |
| `freeze_backbone` | False | Whether to freeze VLM backbone |

### Robot Configuration (train.py line ~160)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `leader_port` | /dev/ttyACM0 | USB port for leader arm |
| `follower_port` | /dev/ttyACM1 | USB port for follower arm |
| `motor_baudrate` | 1000000 | Servo communication speed |
| `camera_fps` | 30 | Camera frame rate |
| `camera_width/height` | 224 | Camera resolution |

---

## Project Structure

```
bh-VLA/
├── train.py                 # Main training script (AMP, early stopping, tqdm)
├── inference.py             # Inference script (real robot + test mode)
├── collect_data.py          # Teleoperation data collection script
├── __init__.py              # Package initialization
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── policies/
│   ├── __init__.py          # Package exports
│   ├── act.py               # ACT policy (ResNet + Transformer, 923 lines)
│   ├── pi05.py              # pi0.5 policy (SigLIP + LLaMA + Flow, 1379 lines)
│   └── config.py            # ACTConfig, Pi05Config, PolicyFactory
├── data/
│   ├── __init__.py          # Package exports
│   ├── dataset.py           # LeRobot/RLDS/Directory/ALOHA datasets
│   ├── transforms.py        # Transforms + augmentation pipeline
│   └── robot_interface.py   # SO-101 + Feetech servo interface
├── checkpoints/             # Saved model checkpoints (auto-created)
│   ├── act_last.pt
│   ├── act_best.pt
│   ├── pi05_last.pt
│   └── pi05_best.pt
├── data/                    # Collected training data (auto-created)
│   └── {task_name}/
│       └── episode_0000/
│           ├── images.npy
│           ├── states.npy
│           ├── actions.npy
│           └── metadata.json
├── outputs/                 # Training outputs (auto-created)
│   ├── action_history.npy   # Inference action history
│   └── logs/                # Training logs
└── logs/                    # Training logs (auto-created)
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "Robot not connected!"

**Solution:** Check USB connections

```bash
# List USB devices
ls /dev/ttyACM*

# Check permissions
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size or use CPU

```bash
python train.py --mode pi05 --batch-size 4 --device cpu
```

### Problem: "Camera not found"

**Solution:** Check camera connection

```bash
# List video devices
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Problem: "Motor communication failed"

**Solution:** Verify motor IDs and baudrates

```bash
# Reset motor configuration
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/ttyACM1
```

---

## License

This project is for research and educational purposes.

## Citation

If you use this code, please cite the original papers:

```bibtex
@misc{zhao2023learningfinemgrainedbimanual,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Tony Z. Zhao and Vikash Kumar and C. Lawrence Zitnick and Sergey Levine and Pamela Abbeel},
  year={2023},
  eprint={2304.13705},
  archivePrefix={arXiv},
}

@misc{physicalintelligence2025pi05vlalearnsexperience,
  title={pi0.5: A Vision-Language-Action Model with Open-World Generalization},
  author={Physical Intelligence team},
  year={2025},
  eprint={2504.16054},
  archivePrefix={arXiv},
}
```

---

## Quick Command Reference

```bash
# Data collection
python collect_data.py --mode act --num-episodes 10

# Training
python train.py --mode act
python train.py --mode pi05

# Inference (real robot)
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt

# Inference (test mode)
python inference.py --mode act --checkpoint ./checkpoints/act_last.pt --test-mode
```
