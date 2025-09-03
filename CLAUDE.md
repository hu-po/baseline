# Keras3 Edge Baseline - Codebase Guide

## Project Overview

This codebase benchmarks Keras 3 backend performance (JAX, PyTorch, TensorFlow) on edge devices to identify optimal configurations for Vision Transformer (ViT) models on CIFAR-100 dataset.

## Core Components

### Main Implementation
- **vit.py**: Complete Vision Transformer implementation with configurable architecture, training loop, and WandB integration
  - Multi-backend support (JAX, PyTorch, TensorFlow)
  - CLI configuration via Tyro
  - Automatic hardware detection
  - Comprehensive metrics tracking

### Target Devices
- **ook**: Intel i7-13620H + NVIDIA RTX 4050 (x86_64)
- **ojo**: NVIDIA Jetson AGX Orin (ARM64) - Current device

## Docker Environments

### Device-Specific Dockerfiles
- `Dockerfile.ook.jax`: JAX backend for x86_64
- `Dockerfile.ook.pytorch`: PyTorch backend for x86_64  
- `Dockerfile.ojo.jax`: JAX backend for ARM64/Jetson
- `Dockerfile.ojo.pytorch`: PyTorch backend for ARM64/Jetson

## Hyperparameter Optimization

### Sweep Scripts
- `run_sweep_ook.sh`: Automated sweeps for ook device
- `run_sweep_ojo.sh`: Automated sweeps for ojo device (Jetson-optimized)
- `sweep_config.json`: WandB sweep configuration

### Key Parameters
- Architecture: projection_dim, num_heads, transformer_layers
- Training: learning_rate, batch_size, weight_decay
- Optimization: Random search with Hyperband early termination

## Configuration

### Environment Setup
- `.env`: Contains WANDB_API_KEY for experiment tracking
- `.claude/settings.local.json`: Claude Code local settings

### Default Values
- Backend: JAX (configurable via --config.keras-backend)
- Batch size: 256 (JAX), 64-128 (PyTorch on Jetson)
- Epochs: 10 (default), 15-20 (sweeps)

## Commands

### Training
```bash
# Basic training with JAX
python vit.py

# Custom configuration
python vit.py --config.keras-backend torch --config.batch-size 64 --config.num-epochs 20

# Without WandB
python vit.py --config.no-use-wandb
```

### Docker Builds
```bash
# Build for current device (ojo/Jetson)
docker build -f Dockerfile.ojo.pytorch -t vit:ojo-pytorch .
docker build -f Dockerfile.ojo.jax -t vit:ojo-jax .
```

### Running Experiments
```bash
# Run with Docker
docker run --rm --runtime nvidia --gpus all -v $PWD:/workspace vit:ojo-pytorch

# Run hyperparameter sweep
./run_sweep_ojo.sh pytorch
```

## Testing & Validation

No explicit test framework is configured. Model validation occurs during training via:
- Validation split (10% by default)
- Top-1 and Top-5 accuracy metrics
- Loss tracking via WandB

## Linting & Type Checking

No linting or type checking commands are currently configured in the project.

## Notes for Development

- First run downloads CIFAR-100 dataset (~170MB)
- Monitor GPU memory with `nvidia-smi` during training
- PyTorch backend recommended for Jetson due to better ARM64 support
- Adjust batch sizes based on available GPU memory
- WandB tracking at: https://wandb.ai/hug/keras3_edge_baseline