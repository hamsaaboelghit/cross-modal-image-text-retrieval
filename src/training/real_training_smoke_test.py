
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
BATCH_SIZE = 2

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / f"fold_{FOLD}_vocab.json"
)

SMOKE_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "smoke_test"
)


def main():
    print("\n" + "=" * 65)
    print("VISIONMATCH AI - TENSORFLOW TRAINING SMOKE TEST")
    print("=" * 65)

    tf.keras.utils.set_random_seed(SEED)

    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found: {VOCAB_PATH}"
        )

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    print(f"Vocabulary size: {len(vocabulary)}")

    train_data = Flickr30kDataset(
        fold_number=FOLD,
        split="train",
    )

    validation_data = Flickr30kDataset(
        fold_number=FOLD,
        split="validation",
    )

    print(f"Training images available: {len(train_data):,}")
    print(f"Validation images available: {len(validation_data):,}")

    # Use only one batch for this smoke test.
    train_dataset = train_data.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        prefetch=False,
    ).take(1)

    validation_dataset = validation_data.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        prefetch=False,
    ).take(1)

    config = TrainingConfig(
        epochs=1,
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
        checkpoint_dir=str(SMOKE_CHECKPOINT_DIR),
        log_dir="runs/smoke_test",
    )

    model = CrossModalRetrievalModel(
        vocab_size=len(vocabulary),
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=False,
        unfreeze_last_block=False,
        use_cross_modal_attention=False,
    )

    loss_fn = MultiPositiveInfoNCELoss(
        temperature=config.temperature,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    print("\nChecking one training batch...")

    train_loss = trainer.train_epoch(train_dataset)

    if not np.isfinite(train_loss):
        raise FloatingPointError(
            f"Invalid training loss: {train_loss}"
        )

    print(f"One-batch training loss: {train_loss:.6f}")
    print("Training step: PASS")

    print("\nChecking one validation batch...")

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    validation_results = evaluator.evaluate(
        validation_dataset
    )

    val_loss = float(validation_results["val_loss"])

    if not np.isfinite(val_loss):
        raise FloatingPointError(
            f"Invalid validation loss: {val_loss}"
        )

    print(f"One-batch validation loss: {val_loss:.6f}")
    print("Validation step: PASS")

    # Check that the model produces valid embeddings.
    sample_batch = next(
        iter(
            validation_data.as_dataset(
                batch_size=BATCH_SIZE,
                shuffle=False,
                prefetch=False,
            ).take(1)
        )
    )

    outputs = model(
        sample_batch["image"],
        sample_batch["captions"],
        training=False,
    )

    image_embeddings = outputs["image_embeddings"]
    text_embeddings = outputs["text_embeddings"]

    tf.debugging.assert_all_finite(
        image_embeddings,
        "Image embeddings contain NaN or Inf.",
    )

    tf.debugging.assert_all_finite(
        text_embeddings,
        "Text embeddings contain NaN or Inf.",
    )

    tf.debugging.assert_near(
        tf.norm(image_embeddings, axis=-1),
        tf.ones((BATCH_SIZE,)),
        atol=1e-3,
        message="Image embeddings are not normalized.",
    )

    print("Image embeddings: PASS")
    print("Text embeddings: PASS")
    print("Embedding normalization: PASS")

    print("\n" + "=" * 65)
    print("ALL SMOKE TEST CHECKS PASSED")
    print("=" * 65)
    print("Only one training batch was used.")
    print("Only one validation batch was used.")
    print("The full dataset was NOT trained.")
    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
