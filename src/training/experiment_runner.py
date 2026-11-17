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
):
    set_seed(seed)

    print("\n" + "=" * 65)
    print(f"VISIONMATCH AI - {experiment_name.upper()}")
    print("=" * 65)

    vocab_path = Path(
        f"data/processed/flickr30k/vocabularies/fold_{fold}_vocab.json"
    )

    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as file:
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
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="none",
        loss_type="info_nce",
        temperature=0.07,
        use_amp=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        seed=seed,
        fold=fold,
        device="auto",
        checkpoint_dir=f"checkpoints/experiments/{experiment_name}",
        log_dir="runs",
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
            output_dim=512,
            dropout=0.1,
        )

    elif model_type == "full":
        model = CrossModalRetrievalModel(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            embedding_dim=300,
            hidden_dim=512,
            num_layers=1,
            dropout=0.2,
            bidirectional=False,
            use_attention=True,
            pretrained=pretrained,
            unfreeze_last_block=unfreeze_last_block,
            use_cross_modal_attention=use_cross_modal_attention,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    print("\nModel configuration:")
    print(f"Model type: {model_type}")
    print(f"Encoder: {encoder_type}")
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

    elapsed_minutes = (time.time() - start_time) / 60

    print("\n" + "=" * 65)
    print(f"{experiment_name.upper()} COMPLETED")
    print("=" * 65)

    if history["train_loss"]:
        print(f"Final training loss: {history['train_loss'][-1]:.6f}")

    if history["val_loss"]:
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {min(history['val_loss']):.6f}")

    print(f"Training time: {elapsed_minutes:.2f} minutes")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"Training log: {logger.get_log_path()}")
    print("Test split was NOT used.")

    return history


if __name__ == "__main__":
    print("Shared TensorFlow experiment runner imported successfully.")