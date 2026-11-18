
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.baseline_model import BaselineRetrievalModel
from src.models.full_model import CrossModalRetrievalModel
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.training.training_logger import TrainingLogger
from src.training.early_stopping import EarlyStopping


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run_experiment(
    experiment_name,
    model_type,
    encoder_type=None,
    use_cross_modal_attention=False,
    pretrained=True,
    unfreeze_last_block=False,
    epochs=5,
    batch_size=8,
    seed=42,
    fold=1,
    learning_rate=1e-4,
    temperature=0.07,
    output_dim=512,
    dropout=None,
    log_dir="runs",
):
    """
    Run one reproducible image-text retrieval experiment.

    Hyperparameter arguments are configurable while their defaults
    preserve the original project configuration.
    """

    if model_type not in {"baseline", "full"}:
        raise ValueError(
            f"Unsupported model_type: {model_type}"
        )

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    if dropout is None:
        dropout = 0.1 if model_type == "baseline" else 0.2

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")

    set_seed(seed)

    print("\n" + "=" * 70)
    print(f"VISIONMATCH AI - {experiment_name.upper()}")
    print("=" * 70)

    vocab_path = Path(
        f"data/processed/flickr30k/vocabularies/fold_{fold}_vocab.json"
    )

    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocab_path}"
        )

    with vocab_path.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)
    print(f"Vocabulary size: {vocab_size}")

    train_data = Flickr30kDataset(fold, "train")
    val_data = Flickr30kDataset(fold, "validation")

    print(f"Training images: {len(train_data):,}")
    print(f"Validation images: {len(val_data):,}")

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="none",
        loss_type="info_nce",
        temperature=temperature,
        use_amp=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        seed=seed,
        fold=fold,
        device="auto",
        checkpoint_dir=f"checkpoints/experiments/{experiment_name}",
        log_dir=log_dir,
        save_best_only=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
    )

    config.display()

    train_dataset = train_data.as_dataset(
        batch_size=config.batch_size,
        shuffle=True,
        drop_remainder=config.drop_last,
        seed=seed,
    )

    val_dataset = val_data.as_dataset(
        batch_size=config.batch_size,
        shuffle=False,
        drop_remainder=False,
        seed=seed,
    )

    if model_type == "baseline":
        model = BaselineRetrievalModel(
            vocab_size=vocab_size,
            embedding_dim=300,
            output_dim=output_dim,
            dropout=dropout,
        )

    else:
        model = CrossModalRetrievalModel(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            embedding_dim=300,
            hidden_dim=512,
            output_dim=output_dim,
            num_layers=1,
            dropout=dropout,
            bidirectional=False,
            use_attention=True,
            pretrained=pretrained,
            unfreeze_last_block=unfreeze_last_block,
            use_cross_modal_attention=use_cross_modal_attention,
        )

    print("\nModel configuration:")
    print(f"Model type: {model_type}")
    print(f"Encoder: {encoder_type}")
    print(f"Retrieval embedding dimension: {output_dim}")
    print(f"Dropout: {dropout}")
    print(f"Learning rate: {learning_rate}")
    print(f"Temperature: {temperature}")
    print(f"Cross-modal attention: {use_cross_modal_attention}")
    print(f"Pretrained ResNet50: {pretrained}")

    loss_fn = MultiPositiveInfoNCELoss(
        temperature=config.temperature,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
    )

    logger = TrainingLogger(
        log_dir=config.log_dir,
        experiment_name=experiment_name,
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    start_time = time.time()

    history = trainer.fit(
        train_loader=train_dataset,
        val_loader=val_dataset,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        early_stopping=early_stopping,
    )

    elapsed_minutes = (time.time() - start_time) / 60.0

    print("\n" + "=" * 70)
    print(f"{experiment_name.upper()} COMPLETED")
    print("=" * 70)

    if history["train_loss"]:
        print(
            f"Final training loss: "
            f"{history['train_loss'][-1]:.6f}"
        )

    if history["val_loss"]:
        print(
            f"Final validation loss: "
            f"{history['val_loss'][-1]:.6f}"
        )
        print(
            f"Best validation loss: "
            f"{min(history['val_loss']):.6f}"
        )

    print(f"Training time: {elapsed_minutes:.2f} minutes")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Training log: {logger.get_log_path()}")
    print("Test split was NOT used.")

    return history


if __name__ == "__main__":
    print(
        "Configurable TensorFlow experiment runner imported successfully."
    )
