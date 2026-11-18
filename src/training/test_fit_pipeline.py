
import csv
import tempfile
from pathlib import Path

import tensorflow as tf

from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.training.training_logger import TrainingLogger
from src.training.early_stopping import EarlyStopping
from src.training.contrastive_loss import MultiPositiveInfoNCELoss


class DummyRetrievalModel(tf.keras.Model):
    """Small TensorFlow model for pipeline testing."""

    def __init__(self, vocab_size=100, embedding_dim=32):
        super().__init__()

        self.image_projection = tf.keras.layers.Dense(
            embedding_dim
        )

        self.text_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=False,
        )

    def call(self, images, captions, training=False):

        image_features = tf.reduce_mean(
            images,
            axis=[1, 2],
        )

        image_embeddings = self.image_projection(
            image_features
        )

        text_features = self.text_embedding(
            captions
        )

        mask = tf.cast(
            tf.not_equal(captions, 0),
            tf.float32,
        )

        mask = mask[..., tf.newaxis]

        text_sum = tf.reduce_sum(
            text_features * mask,
            axis=2,
        )

        token_count = tf.reduce_sum(
            mask,
            axis=2,
        )

        text_embeddings = text_sum / tf.maximum(
            token_count,
            1.0,
        )

        image_embeddings = tf.math.l2_normalize(
            image_embeddings,
            axis=-1,
        )

        text_embeddings = tf.math.l2_normalize(
            text_embeddings,
            axis=-1,
        )

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


def create_dummy_loader(
    num_batches=3,
    batch_size=4,
    num_captions=5,
    sequence_length=12,
):
    """Create synthetic image-caption batches."""

    loader = []

    for _ in range(num_batches):

        images = tf.random.normal(
            (batch_size, 224, 224, 3)
        )

        captions = tf.random.uniform(
            (
                batch_size,
                num_captions,
                sequence_length,
            ),
            minval=1,
            maxval=100,
            dtype=tf.int32,
        )

        captions = tf.concat(
            [
                captions[:, :, :-2],
                tf.zeros(
                    (batch_size, num_captions, 2),
                    dtype=tf.int32,
                ),
            ],
            axis=2,
        )

        loader.append({
            "image": images,
            "captions": captions,
        })

    return loader


def test_fit_pipeline():

    print("\n" + "=" * 60)
    print("VISIONMATCH AI - FULL FIT PIPELINE TEST")
    print("=" * 60)

    tf.keras.utils.set_random_seed(42)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        checkpoint_dir = temp_path / "checkpoints"
        log_dir = temp_path / "logs"

        config = TrainingConfig(
            epochs=3,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            scheduler="cosine",
            loss_type="info_nce",
            temperature=0.07,
            use_amp=False,
            device="cpu",
            gradient_clip_norm=1.0,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
        )

        model = DummyRetrievalModel()

        loss_fn = MultiPositiveInfoNCELoss(
            temperature=config.temperature
        )

        train_loader = create_dummy_loader(
            num_batches=3,
            batch_size=4,
        )

        val_loader = create_dummy_loader(
            num_batches=2,
            batch_size=4,
        )

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir
        )

        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="fit_pipeline_test",
        )

        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
        )

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            config=config,
        )

        initial_weights = [
            weight.copy()
            for weight in model.get_weights()
        ]

        # TEST 1: Complete training pipeline.

        print("\n[Test 1] Complete Training Pipeline")

        history = trainer.fit(
            train_loader=train_loader,
            epochs=3,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            early_stopping=early_stopping,
        )

        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["learning_rate"]) == 3

        print("PASS: Three training epochs completed.")
        print("PASS: Training history is complete.")

        # TEST 2: Model weights updated.

        print("\n[Test 2] Model Parameter Updates")

        final_weights = model.get_weights()

        weights_updated = any(
            not (initial == final).all()
            for initial, final in zip(
                initial_weights,
                final_weights,
            )
        )

        assert weights_updated

        print("PASS: Model weights changed during training.")

        # TEST 3: Checkpoint files.

        print("\n[Test 3] Checkpoint Files")

        latest_checkpoint = tf.train.latest_checkpoint(
            str(checkpoint_dir / "latest")
        )

        best_checkpoint = tf.train.latest_checkpoint(
            str(checkpoint_dir / "best")
        )

        assert latest_checkpoint is not None
        assert best_checkpoint is not None

        assert Path(
            latest_checkpoint + ".index"
        ).is_file()

        assert Path(
            best_checkpoint + ".index"
        ).is_file()

        print("PASS: Latest checkpoint exists.")
        print("PASS: Best checkpoint exists.")

        # TEST 4: Checkpoint metadata.

        print("\n[Test 4] Checkpoint Metadata")

        metadata_path = Path(
            latest_checkpoint + ".json"
        )

        assert metadata_path.is_file()

        import json

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        assert metadata["epoch"] == 2
        assert metadata["global_step"] == 9

        assert tf.math.is_finite(
            metadata["val_loss"]
        )

        print("PASS: Checkpoint metadata is valid.")

        # TEST 5: Early Stopping state.

        print("\n[Test 5] Early Stopping State")

        saved_state = metadata[
            "early_stopping_state"
        ]

        assert saved_state is not None

        assert saved_state["counter"] == (
            early_stopping.counter
        )

        assert saved_state["best_loss"] == (
            early_stopping.best_loss
        )

        assert saved_state["early_stop"] == (
            early_stopping.early_stop
        )

        print("PASS: Early Stopping state saved.")

        # TEST 6: Checkpoint restore.

        print("\n[Test 6] Checkpoint Restore")

        restored_model = DummyRetrievalModel()

        restored_trainer = Trainer(
            model=restored_model,
            loss_fn=MultiPositiveInfoNCELoss(
                temperature=config.temperature
            ),
            config=config,
        )

        # Use a new manager so it tracks the restored model.
        restored_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir
        )

        restore_info = restored_manager.load_checkpoint(
            checkpoint_path=latest_checkpoint,
            model=restored_trainer.model,
            optimizer=restored_trainer.optimizer,
            scheduler=restored_trainer.scheduler,
        )

        assert restore_info["epoch"] == 2
        assert restore_info["global_step"] == 9

        for original, restored in zip(
            model.get_weights(),
            restored_model.get_weights(),
        ):
            tf.debugging.assert_near(
                original,
                restored,
            )

        print("PASS: Checkpoint metadata restored.")
        print("PASS: Model weights restored.")

        # TEST 7: Early Stopping resume.

        print("\n[Test 7] Early Stopping Resume")

        restored_early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
        )

        restored_early_stopping.load_state_dict(
            restore_info["early_stopping_state"]
        )

        assert (
            restored_early_stopping.counter
            == early_stopping.counter
        )

        assert (
            restored_early_stopping.best_loss
            == early_stopping.best_loss
        )

        assert (
            restored_early_stopping.early_stop
            == early_stopping.early_stop
        )

        print("PASS: Early Stopping state restored.")

        # TEST 8: Training CSV log.

        print("\n[Test 8] Training Logger")

        csv_files = list(
            log_dir.glob("*.csv")
        )

        assert len(csv_files) == 1

        with csv_files[0].open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            rows = list(
                csv.DictReader(file)
            )

        assert len(rows) == 3

        required_columns = {
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
            "epoch_duration",
            "timestamp",
        }

        assert required_columns.issubset(
            rows[0].keys()
        )

        print("PASS: CSV log exists.")
        print("PASS: All epochs were logged.")

        # TEST 9: History and progress.

        print("\n[Test 9] History and Training Progress")

        for value in history["train_loss"]:
            assert tf.math.is_finite(value)

        for value in history["val_loss"]:
            assert tf.math.is_finite(value)

        for value in history["learning_rate"]:
            assert value > 0

        assert trainer.global_step == 9
        assert trainer.current_epoch == 3

        print("PASS: All losses are finite.")
        print("PASS: Learning rates are positive.")
        print("PASS: Global step is correct.")
        print("PASS: Epoch count is correct.")

        print("\n" + "=" * 60)
        print("ALL FIT PIPELINE TESTS PASSED!")
        print("=" * 60)

        print(
            "\nNote: Synthetic data was used. "
            "These are not Flickr30k results."
        )


if __name__ == "__main__":
    test_fit_pipeline()
