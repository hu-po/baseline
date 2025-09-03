# Gemini Codebase Understanding

## Project Overview

This project benchmarks the performance of a Vision Transformer (ViT) model using different Keras 3 backends (primarily JAX and PyTorch) on various edge hardware platforms. The goal is to compare training speed, memory usage, and model accuracy to determine the optimal backend for edge deployment.

## Key Components & Technologies

- **Model:** The core model is a Vision Transformer (ViT) implemented in `vit.py`.
- **Backends:** The project is configured to switch between `jax`, `torch`, and `tensorflow` Keras backends.
- **Experiment Tracking:** [Weights & Biases (WandB)](https://wandb.ai/) is heavily integrated for logging metrics and managing hyperparameter sweeps. The `wandb/` directory contains local run data.
- **Containerization:** Docker is used to create reproducible environments for different hardware and backend combinations. Key files include:
    - `Dockerfile.ook.jax` / `Dockerfile.ook.pytorch`
    - `Dockerfile.ojo.jax` / `Dockerfile.ojo.pytorch`

## Hardware Targets: `ook` vs. `ojo`

The project defines two distinct hardware targets:

- **`ook`**: Refers to an x86_64 machine with a standard NVIDIA GPU (e.g., RTX 4050). This is the primary development and testing platform.
- **`ojo`**: Refers to an ARM64-based NVIDIA Jetson AGX Orin, representing a resource-constrained edge device.

Configurations and scripts are often specific to one of these targets (e.g., `run_sweep_ook.sh` vs. `run_sweep_ojo.sh`).

## How to Run Experiments

### Single Runs

Experiments are executed by running the `vit.py` script inside a Docker container. The container provides the correct backend and dependencies.

**Example (JAX on `ook`):**
```bash
docker run --rm --gpus all -v $PWD:/app vit:ook-jax python vit.py --num-epochs=10
```

### Hyperparameter Sweeps

Hyperparameter sweeps are managed by WandB and are the primary method for evaluating different model configurations.

- **Configuration:** `sweep_config.json` defines the search space for hyperparameters.
- **Execution:** The `run_sweep_ook.sh` and `run_sweep_ojo.sh` scripts are used to launch `wandb agent` processes for the respective hardware targets. These scripts will automatically start agents for both JAX and PyTorch backends unless a specific backend is provided as an argument.

**Example (Run sweep on `ook` for JAX only):**
```bash
./run_sweep_ook.sh jax
```
