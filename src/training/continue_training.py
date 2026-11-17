
import json
from pathlib import Path

import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.full_model import CrossModalRetrievalModel
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager


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

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        raise TypeError(
            "Vocabulary must be a dictionary."
        )

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

    print(
        f"Training images: {len(train_data):,}"
    )

    print(
        f"Validation images: {len(validation_data):,}"
    )

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
        log_dir=str(
            PROJECT_ROOT / "runs" / "real_fold1"
        ),
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

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_manager = CheckpointManager(
        str(CHECKPOINT_DIR)
    )

    latest_checkpoint = tf.train.latest_checkpoint(
        str(CHECKPOINT_DIR / "latest")
    )

    if latest_checkpoint is not None:
        resume_from = str(CHECKPOINT_DIR)

        print(
            f"\nCheckpoint found: {latest_checkpoint}"
        )

        print(
            "Training will resume from the saved state."
        )

    else:
        resume_from = None

        print("\nNo checkpoint found.")
        print("Starting a new training run.")

    history = trainer.fit(
        train_loader=train_dataset,
        epochs=TOTAL_EPOCHS,
        val_loader=validation_dataset,
        checkpoint_manager=checkpoint_manager,
        resume_from=resume_from,
    )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

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
        f"Checkpoint directory: {CHECKPOINT_DIR}"
    )

    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
