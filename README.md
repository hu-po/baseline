# Keras3 Edge Baseline 🚀

[![Keras](https://img.shields.io/badge/Keras-3.0%2B-red?logo=keras)](https://keras.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![WandB](https://img.shields.io/badge/WandB-Tracking-yellow?logo=weightsandbiases)](https://wandb.ai/hug/keras3_edge_baseline)

## Overview

This project evaluates Keras 3 backend performance (JAX, PyTorch, TensorFlow) on edge devices to determine optimal compute and memory efficiency. We benchmark Vision Transformer (ViT) models on CIFAR-100 across different hardware configurations.

### Target Hardware

- **ook** - Intel i7-13620H + NVIDIA RTX 4050 (x86_64, CUDA 12.9)
- **ojo** - NVIDIA Jetson AGX Orin (ARM64, CUDA 12.6)

## Prerequisites

- NVIDIA GPU with CUDA 12.x support (`nvidia-smi` shows CUDA Version 12.x)
- NVIDIA Container Toolkit installed for Docker GPU access
- Internet access for downloading CIFAR-100 dataset

## Setup

### 1. WandB Authentication

Create a `.env` file in the project root with your WandB API key:

```bash
# Create .env file
echo "WANDB_API_KEY=your_api_key_here" > .env
```

You can find your API key at: https://wandb.ai/authorize

Alternatively, you can still use environment variables or `wandb login` as before.

#### JAX Backend (GPU)
```bash
docker build -f Dockerfile.ook.jax -t vit:ook-jax .
```

#### PyTorch Backend (GPU)
```bash
docker build -f Dockerfile.ook.pytorch -t vit:ook-pytorch .
```

### 3. Run Experiments

#### Basic Run with JAX
```bash
docker run --rm --gpus all \
  -v $PWD:/app \
  vit:ook-jax
```

#### Basic Run with PyTorch
```bash
docker run --rm --gpus all \
  -v $PWD:/app \
  vit:ook-pytorch
```

#### Advanced Options

Run with custom hyperparameters:
```bash
# Increase epochs and batch size (JAX)
docker run --rm --gpus all \
  -v $PWD:/app \
  vit:ook-jax \
  python vit.py --num-epochs 100 --batch-size 512

# Custom run name and tags (PyTorch)
docker run --rm --gpus all \
  -v $PWD:/app \
  vit:ook-pytorch \
  python vit.py \
    --wandb-run-name "pytorch-rtx4050-100ep" \
    --wandb-tags '["baseline", "100-epochs", "ook"]'
```

#### Run without WandB
```bash
docker run --rm --gpus all \
  -v $PWD:/app \
  vit:ook-jax \
  python vit.py --no-use-wandb
```

## Configuration

The ViT model can be configured via command-line arguments:

### Model Architecture
- `--image-size`: Input image size (default: 72)
- `--patch-size`: Size of image patches (default: 6)
- `--projection-dim`: Projection dimension (default: 64)
- `--num-heads`: Number of attention heads (default: 4)
- `--transformer-layers`: Number of transformer blocks (default: 8)

### Training Parameters
- `--learning-rate`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay (default: 0.0001)
- `--batch-size`: Batch size (default: 256 for JAX, 64 recommended for PyTorch)
- `--num-epochs`: Number of epochs (default: 10)
- `--validation-split`: Validation split ratio (default: 0.1)

### Backend Selection
- `--keras-backend`: Choose backend: "jax", "torch", or "tensorflow"

### WandB Settings
- `--use-wandb` / `--no-use-wandb`: Enable/disable WandB logging (default: True)
- `--wandb-project`: WandB project name (default: "keras3_edge_baseline")
- `--wandb-entity`: WandB entity/team (default: "hug")
- `--wandb-run-name`: Custom run name
- `--wandb-tags`: List of tags for the run

## Monitoring

Track experiments in real-time at: https://wandb.ai/hug/keras3_edge_baseline/workspace

Key metrics tracked:
- Training/validation accuracy and loss
- Top-5 accuracy
- Model parameters count
- Training time per epoch
- GPU memory usage

## Project Structure

```
baseline/
├── vit.py                    # Main ViT implementation with all features
├── Dockerfile.ook.jax        # JAX backend container for x86_64
├── Dockerfile.ook.pytorch    # PyTorch backend container for x86_64
├── system_ook.md            # Hardware specs for ook (x86_64)
├── system_ojo.md            # Hardware specs for ojo (ARM64)
├── .env                     # WandB API key (create this file)
└── README.md                # This file
```

## Performance Comparison

The goal is to identify the optimal Keras 3 backend for edge deployment by comparing:

1. **Training Speed** - Time per epoch and total training time
2. **Memory Usage** - GPU memory consumption during training/inference
3. **Model Accuracy** - Final test accuracy on CIFAR-100
4. **Resource Efficiency** - CPU/GPU utilization patterns
5. **Cross-platform Compatibility** - Performance on x86_64 vs ARM64

- Monitor GPU memory with `nvidia-smi` while running
- The first run downloads CIFAR-100 dataset (~170MB)

## Troubleshooting

### WandB Authentication Issues
```bash
# Check if .env file exists and has correct key
cat .env

# Or set manually
docker run --rm --gpus all \
  -v $PWD:/app \
  -e WANDB_API_KEY=your_key_here \
  vit:ook-jax
```

### GPU Not Detected
Verify NVIDIA Container Toolkit:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

## Hyperparameter Sweeps

The project includes automated hyperparameter optimization using WandB sweeps to find the best model configurations for each backend and device.

### Quick Start Sweeps

```bash
# Run sweeps for both JAX and PyTorch backends
./run_sweep_ook.sh

# Run specific backend only
./run_sweep_ojo.sh pytorch
./run_sweep_ojo.sh jax
```

### Sweep Configuration

The sweep optimizes the following hyperparameters:

**Architecture Parameters:**
- `projection_dim`
- `num_heads`
- `transformer_layers`
- `image_size`
- `patch_size`

**Training Parameters:**
- `learning_rate`
- `batch_size`
- `weight_decay`
- `num_epochs`

**Optimization:**
- **Method**: Random search with Hyperband early termination
- **Metric**: Maximize `val_top-5-accuracy`
- **Backend**: Compares JAX vs PyTorch performance

### Monitoring Sweeps

- **Dashboard**: https://wandb.ai/hug/keras3_edge_baseline/sweeps
- **Live metrics**: Training loss, accuracy, validation metrics
- **Comparison**: Side-by-side backend and device performance
- **Resource usage**: GPU memory, training time per epoch