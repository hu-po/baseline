"""
Vision Transformer (ViT) implementation for image classification on CIFAR-100.
- https://arxiv.org/abs/2010.11929
- https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import keras
import numpy as np
import tyro
from keras import layers, ops

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads the .env file from the current directory
except ImportError:
    pass  # dotenv not installed, continue without it

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer model and training."""
    # Model architecture
    num_classes: int = 100
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    image_size: int = 72
    patch_size: int = 6
    projection_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 8
    transformer_units: List[int] = None
    mlp_head_units: List[int] = None
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 256
    num_epochs: int = 10
    validation_split: float = 0.1
    
    # Other settings
    keras_backend: str = "jax"
    checkpoint_path: str = "/tmp/checkpoint.weights.h5"
    node_name: str = "unknown"  # Will be auto-detected or set via CLI
    
    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "keras3_edge_baseline"
    wandb_entity: Optional[str] = "hug"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.transformer_units is None:
            self.transformer_units = [self.projection_dim * 2, self.projection_dim]
        if self.mlp_head_units is None:
            self.mlp_head_units = [2048, 1024]
        
        self.num_patches = (self.image_size // self.patch_size) ** 2


def create_data_augmentation(config: ViTConfig, x_train: np.ndarray) -> keras.Sequential:
    """Create data augmentation pipeline."""
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(config.image_size, config.image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    data_augmentation.layers[0].adapt(x_train)
    return data_augmentation


def mlp(x, hidden_units: List[int], dropout_rate: float):
    """Multi-layer perceptron block."""
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class CustomWandbCallback(keras.callbacks.Callback):
    """Custom WandB callback that works with all Keras backends."""
    
    def __init__(self):
        super().__init__()
        self.wandb_available = False
        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            pass
    
    def on_epoch_end(self, epoch, logs=None):
        if not self.wandb_available or logs is None:
            return
        
        # Log all metrics from the epoch
        wandb_logs = {}
        for key, value in logs.items():
            # Convert numpy types to Python types for wandb
            if hasattr(value, 'item'):
                value = value.item()
            elif hasattr(value, '__float__'):
                value = float(value)
            wandb_logs[key] = value
        
        # Add epoch number
        wandb_logs['epoch'] = epoch + 1
        
        try:
            self.wandb.log(wandb_logs)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def on_train_end(self, logs=None):
        if not self.wandb_available:
            return
        
        if logs:
            final_logs = {}
            for key, value in logs.items():
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, '__float__'):
                    value = float(value)
                final_logs[f"final_{key}"] = value
            
            try:
                self.wandb.log(final_logs)
            except Exception as e:
                logger.warning(f"Failed to log final metrics to wandb: {e}")


class Patches(layers.Layer):
    """Extract patches from images."""
    
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        # Ensure float dtype for JAX/cuDNN compatibility
        images = ops.convert_to_tensor(images, dtype="float32")
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    """Encode patches with linear projection and position embeddings."""
    
    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def create_vit_classifier(config: ViTConfig, data_augmentation: keras.Sequential) -> keras.Model:
    """Build the Vision Transformer model."""
    inputs = keras.Input(shape=config.input_shape)
    
    # Augment data
    augmented = data_augmentation(inputs)
    
    # Create and encode patches
    patches = Patches(config.patch_size)(augmented)
    encoded_patches = PatchEncoder(config.num_patches, config.projection_dim)(patches)

    # Transformer blocks
    for _ in range(config.transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=config.num_heads, key_dim=config.projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=config.transformer_units, dropout_rate=0.1)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Final representation
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Classification head
    features = mlp(representation, hidden_units=config.mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(config.num_classes)(features)
    
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model: keras.Model, config: ViTConfig, x_train: np.ndarray, 
                  y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    """Compile, train, and evaluate the model."""
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    ]
    
    # Add custom wandb callback if enabled
    if config.use_wandb:
        callbacks.append(CustomWandbCallback())

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config.batch_size,
        epochs=config.num_epochs,
        validation_split=config.validation_split,
        callbacks=callbacks,
    )

    model.load_weights(config.checkpoint_path)
    _, test_accuracy, test_top5_accuracy = model.evaluate(x_test, y_test)
    
    logger.info(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    logger.info(f"Test top 5 accuracy: {round(test_top5_accuracy * 100, 2)}%")
    
    # Log final test metrics to wandb
    if config.use_wandb:
        try:
            import wandb
            wandb.log({
                "test_accuracy": test_accuracy,
                "test_top5_accuracy": test_top5_accuracy,
            })
        except Exception as e:
            logger.warning(f"wandb logging failed ({e}).")

    return history, test_accuracy, test_top5_accuracy


def main(config: ViTConfig = ViTConfig()):
    """Main training function."""
    # Auto-detect node name if not set
    if config.node_name == "unknown":
        import socket
        config.node_name = socket.gethostname()
    
    # Set Keras backend
    os.environ["KERAS_BACKEND"] = config.keras_backend
    
    # Initialize wandb if enabled
    if config.use_wandb:
        try:
            import wandb
        except Exception as e:
            logger.warning(f"wandb not available ({e}); disabling wandb.")
            config.use_wandb = False
            wandb = None  # type: ignore

    if config.use_wandb:
        wandb_config = {
            # System info
            "backend": config.keras_backend,
            "node": config.node_name,
            "architecture": "Vision Transformer",
            "dataset": "CIFAR-100",
            
            # Model architecture
            "num_classes": config.num_classes,
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "projection_dim": config.projection_dim,
            "num_heads": config.num_heads,
            "transformer_layers": config.transformer_layers,
            "transformer_units": config.transformer_units,
            "mlp_head_units": config.mlp_head_units,
            "num_patches": config.num_patches,
            
            # Training parameters
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "validation_split": config.validation_split,
        }
        
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            tags=config.wandb_tags,
            config=wandb_config,
        )
        logger.info(f"Wandb run initialized: {wandb.run.name}")
    
    # Load CIFAR-100 dataset
    logger.info("Loading CIFAR-100 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    
    logger.info(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    logger.info(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
    
    # Create data augmentation
    logger.info("Setting up data augmentation...")
    data_augmentation = create_data_augmentation(config, x_train)
    
    # Build model
    logger.info("Building Vision Transformer model...")
    logger.info(f"Image size: {config.image_size} x {config.image_size}")
    logger.info(f"Patch size: {config.patch_size} x {config.patch_size}")
    logger.info(f"Patches per image: {config.num_patches}")
    
    vit_classifier = create_vit_classifier(config, data_augmentation)
    
    # Log model summary to wandb
    if config.use_wandb:
        try:
            import wandb
            model_params = vit_classifier.count_params()
            wandb.log({"total_parameters": model_params})
            logger.info(f"Total model parameters: {model_params:,}")
        except Exception as e:
            logger.warning(f"wandb parameter logging failed ({e}).")
    
    # Train and evaluate
    logger.info("Starting training...")
    history, test_accuracy, test_top5_accuracy = run_experiment(
        vit_classifier, config, x_train, y_train, x_test, y_test
    )
    
    # Log final summary
    if config.use_wandb:
        try:
            import wandb
            wandb.summary["final_test_accuracy"] = test_accuracy
            wandb.summary["final_test_top5_accuracy"] = test_top5_accuracy
            wandb.finish()
        except Exception as e:
            logger.warning(f"wandb summary logging failed ({e}).")
    
    logger.info("Training complete!")
    return history


if __name__ == "__main__":
    tyro.cli(main)
