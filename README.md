# Vision Transformer (ViT) – Docker GPU Runbook

This repo contains a ViT (Vision Transformer) example adapted to run with Keras 3 on the JAX backend. A GPU‑only Docker image is provided for reproducible runs on systems with NVIDIA GPUs.

## Requirements
- NVIDIA GPU + recent driver (CUDA 12.x capable). Host shows `CUDA Version: 12.x` in `nvidia-smi`.
- NVIDIA Container Toolkit installed on the host so Docker can access the GPU.
- Internet access in the container (to download CIFAR‑100 on first run).

## Build (GPU‑only)
- Build the image from the GPU Dockerfile:
  - `docker build -f Dockerfile.ook.jax -t vit:ook-jax .`

Notes:
- The image pins `jax[cuda12_pip]==0.4.31` for CUDA 12 support and uses base `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`.
- `KERAS_BACKEND=jax` and `JAX_PLATFORM_NAME=gpu` are set by default in the image.

## Quick Test (1 epoch, no W&B)
Run a short smoke test to verify GPU and dependencies:
- `docker run --rm --gpus all -e KERAS_BACKEND=jax -e WANDB_MODE=offline vit:ook-jax bash -lc "python vit.py --config.num-epochs 1 --config.batch-size 64 --config.no-use-wandb"`

You should see CIFAR‑100 download (first run), training start, and evaluation after 1 epoch.

## Full Run (with W&B optional)
- Without Weights & Biases logging:
  - `docker run --rm --gpus all -e KERAS_BACKEND=jax vit:ook-jax bash -lc "python vit.py --config.num-epochs 10 --config.batch-size 256 --config.no-use-wandb"`
- With Weights & Biases logging (requires an API key):
  - `docker run --rm --gpus all -e KERAS_BACKEND=jax -e WANDB_API_KEY=$WANDB_API_KEY vit:ook-jax bash -lc "python vit.py --config.num-epochs 10 --config.batch-size 256 --config.use-wandb"`

## Using Local Code Without Rebuilds
By default the image contains a copy of `vit.py` at build time. If you are actively editing `vit.py`, mount the workspace to use your local changes without rebuilding:
- `docker run --rm --gpus all -v $PWD:/app -w /app -e KERAS_BACKEND=jax vit:ook-jax bash -lc "python vit.py --config.num-epochs 1 --config.batch-size 64 --config.no-use-wandb"`

## Troubleshooting
- No GPU detected / JAX shows CPU devices:
  - Ensure Docker is run with `--gpus all` and NVIDIA Container Toolkit is installed.
  - Confirm `nvidia-smi` works on the host and inside the container (`docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi`).
- W&B import errors:
  - The script gracefully disables W&B if it’s unavailable. Use `--config.no-use-wandb` or set `WANDB_MODE=offline` for local tests.
- CIFAR‑100 download stalls:
  - Verify the container has outbound network access and try again.

## File Overview
- `Dockerfile.ook.jax`: GPU‑only Dockerfile (CUDA 12, JAX backend).
- `vit.py`: ViT training script using Keras 3 + JAX with tyro CLI (`--config.*` flags).

