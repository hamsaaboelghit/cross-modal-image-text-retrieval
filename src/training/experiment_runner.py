
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
from src.training.margin_loss import MarginBasedContrastiveLoss
from src.training.triplet_loss import TripletContrastiveLoss
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
    train_all_backbone=False,
    caption_count=5,
    image_augmentation="none",
    negative_sampling_strategy="random",
    epochs=5,
    batch_size=8,
    seed=42,
    fold=1,
    learning_rate=1e-4,
    temperature=0.07,
    loss_type="info_nce",
    margin=None,
    drop_last=False,
    output_dim=512,
    dropout=None,
    log_dir="runs",
    max_train_steps_per_epoch=None,
    max_validation_steps=None,
    resume_from=None,
    resume_skip_batches=0,
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

    if loss_type not in {"info_nce", "margin", "triplet"}:
        raise ValueError("loss_type must be one of: info_nce, margin, triplet.")
    if margin is None:
        margin = 1.0 if loss_type == "margin" else 0.2
    if margin <= 0 or margin > 2:
        raise ValueError("margin must be greater than 0 and at most 2.")

    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    if dropout is None:
        # Keep the dropout rate fixed across the architecture comparison.
        dropout = 0.1

    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")

    if (
        isinstance(resume_skip_batches, bool)
        or not isinstance(resume_skip_batches, int)
        or resume_skip_batches < 0
    ):
        raise ValueError("resume_skip_batches must be a non-negative integer.")

    if resume_skip_batches and resume_from is None:
        raise ValueError("resume_from is required when skipping prior batches.")

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

    train_data = Flickr30kDataset(
        fold,
        "train",
        caption_count=caption_count,
        caption_seed=seed,
        image_augmentation=image_augmentation,
    )
    # Keep all five reference captions for validation regardless of the
    # training caption ablation, so evaluation remains comparable.
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
        loss_type=loss_type,
        temperature=temperature,
        margin=margin,
        negative_sampling_strategy=negative_sampling_strategy,
        use_amp=False,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
        max_train_steps_per_epoch=max_train_steps_per_epoch,
        max_validation_steps=max_validation_steps,
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

    # On a resumed partial epoch, rebuild the deterministic seeded dataset
    # and skip batches already consumed by the previous process.
    if resume_skip_batches:
        print(f"Skipping {resume_skip_batches} previously consumed batches.")
        train_dataset = train_dataset.skip(resume_skip_batches)

    val_dataset = val_data.as_dataset(
        batch_size=config.batch_size,
        shuffle=False,
        drop_remainder=config.drop_last,
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
            train_all_backbone=train_all_backbone,
            use_cross_modal_attention=use_cross_modal_attention,
        )

    print("\nModel configuration:")
    print(f"Model type: {model_type}")
    print(f"Encoder: {encoder_type}")
    print(f"Retrieval embedding dimension: {output_dim}")
    print(f"Dropout: {dropout}")
    print(f"Learning rate: {learning_rate}")
    print(f"Temperature: {temperature}")
    print(f"Loss function: {loss_type}")
    print(f"Loss margin: {margin if loss_type != 'info_nce' else 'N/A'}")
    print(f"Cross-modal attention: {use_cross_modal_attention}")
    print(f"Pretrained ResNet50: {pretrained}")
    print(f"Train all ResNet layers: {train_all_backbone}")
    print(f"Training captions per image: {caption_count}")
    print(f"Training image augmentation: {image_augmentation}")
    print(f"Negative sampling strategy: {negative_sampling_strategy}")
    print(f"Train steps per epoch: {max_train_steps_per_epoch or 'FULL DATASET'}")
    print(f"Validation batches per epoch: {max_validation_steps or 'FULL DATASET'}")

    if loss_type == "info_nce":
        loss_fn = MultiPositiveInfoNCELoss(temperature=config.temperature)
    elif loss_type == "margin":
        loss_fn = MarginBasedContrastiveLoss(margin=config.margin)
    else:
        loss_fn = TripletContrastiveLoss(
            margin=config.margin,
            negative_sampling_strategy=config.negative_sampling_strategy,
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
        resume_from=resume_from,
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
