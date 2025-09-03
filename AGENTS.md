# Agents Guide: Keras3 Edge Baseline

This document orients agents and contributors to the repository layout, main entry points, runtime expectations, and how to run/extend experiments. It complements README.md with implementation details and gotchas relevant to automation.

## What This Repo Does
- Benchmarks Keras 3 backends (JAX, PyTorch; TensorFlow optional) on edge hardware using a Vision Transformer (ViT) trained on CIFAR-100.
- Logs metrics and runs optional hyperparameter sweeps via Weights & Biases (WandB).
- Provides Docker images and scripts for two hardware targets:
  - `ook` (x86_64, RTX 4050, CUDA 12.x)
  - `ojo` (ARM64, Jetson AGX Orin, CUDA 12.6)

## Repository Map
- `vit.py` — Single-file ViT training entrypoint with CLI, backend selection, model, training loop, and WandB integration.
- `Dockerfile.ook.jax` / `Dockerfile.ook.pytorch` — x86_64 (RTX 4050) images for JAX / PyTorch backends.
- `Dockerfile.ojo.jax` / `Dockerfile.ojo.pytorch` — Jetson (ARM64) images for JAX / PyTorch backends.
- `run_sweep_ook.sh` / `run_sweep_ojo.sh` — Helpers to create and run WandB sweeps per device/backend.
- `sweep_config.json` — WandB sweep search space; scripts augment it with device/backend.
- `system_ook.md` / `system_ojo.md` — Hardware specs for both targets.
- `README.md` — How to build/run under Docker and tune hyperparameters.
- `GEMINI.md` — Brief codebase overview for LLMs/agents.
- `.env` / `.env.example` — WandB credentials and settings (optional).

## Primary Entry Point: `vit.py`
`vit.py` is designed to run end-to-end training and evaluation with a consistent CLI across backends.

Key components:
- Backend selection before importing Keras:
  - `_set_backend_from_env_or_cli(sys.argv)` sets `KERAS_BACKEND` from `--keras-backend` (or `--keras_backend`) unless already set in env; defaults to `"jax"`.
  - Keras is imported only after the backend is finalized.
- Configuration: `ViTConfig` dataclass defines model/training/WandB fields. It also computes `num_patches` and initializes sensible defaults for transformer/MLP units.
- Data pipeline: `create_data_augmentation` builds a Keras `Sequential` with normalization, resize, flip, rotation, and zoom (adapts normalization to training data).
- Model: `Patches` and `PatchEncoder` layers, transformer blocks with MHA + MLP, and a classifier head (`create_vit_classifier`).
- Training: `run_experiment` compiles with `AdamW`, tracks accuracy and top-5 accuracy, checkpoints best weights, and evaluates on test set.
- WandB: `CustomWandbCallback` logs epoch metrics cross-backend; `main()` optionally initializes WandB (project/entity/name/tags, params, summaries).
- Dataset: downloads/loads CIFAR-100 from `keras.datasets` on first run (requires network).
- CLI: uses `tyro` to parse flags directly into `ViTConfig`. A small shim normalizes underscores to dashes for compatibility with some tooling.

Outputs/logging:
- Console logs for shapes, config, training progress, and final test metrics.
- Optional WandB: metrics per epoch, parameter count, final test metrics, and run summaries.
- Best weights saved to `ViTConfig.checkpoint_path` (default `/tmp/checkpoint.weights.h5`).

## Command-Line Interface
`vit.py` accepts top-level flags that mirror fields in `ViTConfig`. Examples:
- Backend: `--keras-backend jax|torch|tensorflow`
- Training: `--num-epochs 20 --batch-size 256 --learning-rate 0.001 --weight-decay 0.0001`
- Model: `--image-size 72 --patch-size 6 --projection-dim 64 --num-heads 4 --transformer-layers 8`
- WandB: `--use-wandb` (default) or `--no-use-wandb`, plus `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, `--wandb-tags '["tag1","tag2"]'`

Important notes for agents:
- The code expects top-level flags (not `--config.*`). Some docs/examples may show `--config.<field>`—those won’t be parsed by `vit.py` as-is.
- Flags accept both `--kebab-case` and `--under_score` forms (underscores get normalized to dashes).
- If `KERAS_BACKEND` is already set in the environment, it takes precedence over CLI until `main()` reasserts `config.keras_backend`.

## Running
- Local (system must have the chosen backend and GPU stack installed):
  - `python vit.py --keras-backend jax --num-epochs 10`
  - `python vit.py --keras-backend torch --batch-size 64`
- Docker (recommended):
  - Build (ook): `docker build -f Dockerfile.ook.jax -t vit:ook-jax .`
  - Run (GPU): `docker run --rm --gpus all -v $PWD:/app vit:ook-jax python vit.py --num-epochs 10`
  - Build (ook, PyTorch): `docker build -f Dockerfile.ook.pytorch -t vit:ook-pytorch .`
  - Run: `docker run --rm --gpus all -v $PWD:/app vit:ook-pytorch python vit.py --keras-backend torch`
  - Jetson images (`Dockerfile.ojo.*`) set appropriate bases/env for ARM64.

WandB setup:
- Provide `WANDB_API_KEY` via `.env`, environment variable, or `wandb login` inside the container.
- Optional `WANDB_PROJECT` and `WANDB_ENTITY` control where runs/sweeps are logged.
- Disable logging with `--no-use-wandb`.

## Hyperparameter Sweeps
- Sweep space: `sweep_config.json` (learning_rate, batch_size, projection_dim, num_heads, transformer_layers, image_size, patch_size, weight_decay, num_epochs, keras_backend).
- Agents/scripts automatically set `keras_backend` and `node_name` and name the sweep accordingly.
- Launchers:
  - `./run_sweep_ook.sh [jax|pytorch|both]`
  - `./run_sweep_ojo.sh [jax|pytorch|both]`
- What they do:
  1) Build the matching Docker image.
  2) Create the WandB sweep (requires `WANDB_ENTITY`).
  3) Start a `wandb agent` in the container with GPU access.
- Output: Links to the sweep dashboard and continuous metric streaming.

## Backends and Environment
- JAX
  - x86_64: CUDA12 wheels installed in `Dockerfile.ook.jax` and `JAX_PLATFORM_NAME=gpu`.
  - Jetson: Uses `dustynv/jax` base with CUDA/cuDNN stack preconfigured.
- PyTorch
  - x86_64: Installs `torch`/`torchvision` cu121 wheels in `Dockerfile.ook.pytorch`.
  - Jetson: Uses `dustynv/l4t-pytorch` base image.
- TensorFlow
  - Not explicitly provisioned in Dockerfiles; supported by Keras 3 if installed in your environment and selected via `--keras-backend tensorflow`.

## Common Tasks (for Agents)
- Train a quick baseline (JAX): `docker run --rm --gpus all -v $PWD:/app vit:ook-jax python vit.py --num-epochs 5`
- Switch backend (PyTorch): `docker run --rm --gpus all -v $PWD:/app vit:ook-pytorch python vit.py --keras-backend torch --batch-size 64`
- Disable WandB: add `--no-use-wandb` (still trains/evaluates, skips remote logging).
- Change architecture: set `--projection-dim`, `--num-heads`, `--transformer-layers`, `--image-size`, `--patch-size`.
- Check parameters: Keras `model.count_params()` is logged to console and WandB (if enabled).

## Extending The Codebase
- New datasets: replace CIFAR-100 load with your dataset; keep `(x_train, y_train), (x_test, y_test)` and shapes consistent with `ViTConfig.input_shape`.
- Additional metrics: add Keras metrics in `model.compile(...)` and they’ll be auto-logged by the custom WandB callback.
- Alternate optimizers/schedules: change `optimizer` in `run_experiment`.
- New backends: ensure the backend is supported by Keras 3 in your environment and pass `--keras-backend <name>`; add/adjust a Dockerfile if containerized.
- Memory/throughput: tune `batch_size`, `image_size`, and transformer depth/width; JAX usually supports larger batches on desktop GPUs.

## Gotchas & Tips
- Set backend before importing Keras: the code already enforces this, but external wrappers launching `vit.py` should avoid importing Keras first.
- CLI flag shape: use top-level flags (e.g., `--num-epochs`), not `--config.*`.
- Jetson specifics: prefer the PyTorch backend on Jetson; use smaller batch sizes and reduced model complexity.
- First run downloads CIFAR-100 (~170MB) and needs network access.
- Checkpoint path defaults to `/tmp/...`; use `--checkpoint-path` to persist in a mounted folder when running in containers.
- WandB sweeps require `WANDB_ENTITY` (user/team); `.env` is supported by helper scripts.

## Related Files
- Hardware specs: `system_ook.md`, `system_ojo.md`.
- Human notes: `HUMANS.md`.
- Additional overview: `GEMINI.md`.

---
If you want me to add examples for TensorFlow runs, or wire up a minimal TensorFlow Dockerfile, say the word and I’ll draft it.

