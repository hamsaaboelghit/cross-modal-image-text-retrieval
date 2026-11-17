
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.full_model import CrossModalRetrievalModel
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOLD = 1
SEED = 42
BATCH_SIZE = 8
TOTAL_EPOCHS = 5

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / f"fold_{FOLD}_vocab.json"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "real_fold1"
)


def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)


def load_vocabulary():
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found: {VOCAB_PATH}"
        )

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        raise TypeError("Vocabulary must be a dictionary.")

    return vocabulary


def build_model(vocab_size):
    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=True,
        unfreeze_last_block=False,
        use_cross_modal_attention=False,
    )

    # Create all model variables before checkpoint restoration.
    dummy_images = tf.zeros(
        (1, 224, 224, 3),
        dtype=tf.float32,
    )

    dummy_captions = tf.zeros(
        (1, 5, 34),
        dtype=tf.int32,
    )

    model(
        dummy_images,
        dummy_captions,
        training=False,
    )

    return model


def main():
    print("\n" + "=" * 65)
    print("VISIONMATCH AI - CONTINUE TRAINING")
    print("=" * 65)

    set_seed(SEED)

    vocabulary = load_vocabulary()
    vocab_size = len(vocabulary)

    print(f"Vocabulary size: {vocab_size}")

    train_data = Flickr30kDataset(
        fold_number=FOLD,
        split="train",
    )

    validation_data = Flickr30kDataset(
        fold_number=FOLD,
        split="validation",
    )

    print(f"Training images: {len(train_data):,}")
    print(f"Validation images: {len(validation_data):,}")

    train_dataset = train_data.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        prefetch=True,
    )

    validation_dataset = validation_data.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        prefetch=True,
    )

    config = TrainingConfig(
        epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="none",
        loss_type="info_nce",
        temperature=0.07,
        use_amp=False,
        seed=SEED,
        fold=FOLD,
        device="auto",
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir="runs/real_fold1",
        save_best_only=True,
    )

    config.display()

    model = build_model(vocab_size)

    loss_fn = MultiPositiveInfoNCELoss(
        temperature=config.temperature,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the checkpoint root names consistent with evaluation_runner.py.
    epoch_variable = tf.Variable(
        0,
        dtype=tf.int64,
        trainable=False,
        name="epoch",
    )

    global_step_variable = tf.Variable(
        0,
        dtype=tf.int64,
        trainable=False,
        name="global_step",
    )

    best_val_loss_variable = tf.Variable(
        float("inf"),
        dtype=tf.float32,
        trainable=False,
        name="best_val_loss",
    )

    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=trainer.optimizer,
        epoch=epoch_variable,
        global_step=global_step_variable,
        best_val_loss=best_val_loss_variable,
    )

    latest_manager = tf.train.CheckpointManager(
        checkpoint,
        str(CHECKPOINT_DIR),
        max_to_keep=3,
        checkpoint_name="ckpt",
    )

    best_manager = tf.train.CheckpointManager(
        checkpoint,
        str(CHECKPOINT_DIR / "best"),
        max_to_keep=1,
        checkpoint_name="best",
    )

    latest_path = latest_manager.latest_checkpoint

    if latest_path:
        print(f"\nRestoring training state: {latest_path}")

        status = checkpoint.restore(latest_path)
        status.expect_partial()

        start_epoch = int(epoch_variable.numpy())
        best_val_loss = float(best_val_loss_variable.numpy())

        trainer.global_step = int(
            global_step_variable.numpy()
        )

        print(f"Resuming after epoch: {start_epoch}")
        print(f"Previous best validation loss: {best_val_loss:.6f}")
        print(f"Restored global step: {trainer.global_step}")

    else:
        start_epoch = 0
        best_val_loss = float("inf")

        print("\nNo TensorFlow checkpoint found.")
        print("Starting a new training run.")

    if start_epoch >= TOTAL_EPOCHS:
        print(
            f"\nTraining already reached {TOTAL_EPOCHS} epochs."
        )
        return

    print("\nStarting training...")

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print("\n" + "=" * 65)
        print(f"EPOCH {epoch + 1}/{TOTAL_EPOCHS}")
        print("=" * 65)

        train_loss = trainer.train_epoch(train_dataset)

        validation_results = evaluator.evaluate(
            validation_dataset
        )

        val_loss = float(validation_results["val_loss"])

        if not np.isfinite(train_loss):
            raise FloatingPointError(
                f"Non-finite training loss at epoch {epoch + 1}."
            )

        if not np.isfinite(val_loss):
            raise FloatingPointError(
                f"Non-finite validation loss at epoch {epoch + 1}."
            )

        print(f"\nTraining loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")

        epoch_variable.assign(epoch + 1)
        global_step_variable.assign(trainer.global_step)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_val_loss_variable.assign(best_val_loss)

        latest_path = latest_manager.save(
            checkpoint_number=epoch + 1
        )

        print(f"Latest checkpoint saved: {latest_path}")

        if improved:
            best_path = best_manager.save(
                checkpoint_number=epoch + 1
            )

            print(f"New best checkpoint saved: {best_path}")

        print(f"Global step: {trainer.global_step}")
        print(f"Best validation loss: {best_val_loss:.6f}")

    print("\n" + "=" * 65)
    print("CONTINUED TRAINING COMPLETED")
    print("=" * 65)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
