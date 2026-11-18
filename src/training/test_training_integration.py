
import tensorflow as tf

from src.training.train_config import TrainingConfig
from src.training.optimizer import (
    build_optimizer,
    build_scheduler,
)
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.training.contrastive_loss import MultiPositiveInfoNCELoss


class DummyRetrievalModel(tf.keras.Model):
    """Small model for training integration tests."""

    def __init__(self, vocab_size=100, embedding_dim=32):
        super().__init__()

        self.image_projection = tf.keras.layers.Dense(
            embedding_dim
        )

        self.text_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
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
        )[..., tf.newaxis]

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

        return {
            "image_embeddings": tf.math.l2_normalize(
                image_embeddings,
                axis=-1,
            ),
            "text_embeddings": tf.math.l2_normalize(
                text_embeddings,
                axis=-1,
            ),
        }


def create_dummy_loader(
    num_batches=3,
    batch_size=4,
    num_captions=5,
    sequence_length=12,
):

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


def test_training_integration():

    print("\n" + "=" * 55)
    print("VISIONMATCH AI - TRAINING INTEGRATION TESTS")
    print("=" * 55)

    tf.keras.utils.set_random_seed(42)

    config = TrainingConfig(
        epochs=2,
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
    )

    model = DummyRetrievalModel()

    # Build model before creating optimizer.
    model(
        tf.zeros((1, 224, 224, 3)),
        tf.ones((1, 5, 12), dtype=tf.int32),
        training=False,
    )

    loss_fn = MultiPositiveInfoNCELoss(
        temperature=config.temperature
    )

    train_loader = create_dummy_loader()

    val_loader = create_dummy_loader(
        num_batches=2,
    )

    # TEST 1: Optimizer creation.

    print("\n[Test 1] Optimizer Creation")

    optimizer = build_optimizer(
        model,
        config,
    )

    assert isinstance(
        optimizer,
        tf.keras.optimizers.AdamW,
    )

    print("PASS: AdamW optimizer created.")

    # TEST 2: Scheduler creation.

    print("\n[Test 2] Scheduler Creation")

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        tf.keras.callbacks.LearningRateScheduler,
    )

    print("PASS: Cosine scheduler created.")

    # TEST 3: Trainer initialization.

    print("\n[Test 3] Trainer Initialization")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert trainer.optimizer is optimizer
    assert trainer.scheduler is scheduler

    print("PASS: Trainer initialized.")
    print("PASS: Optimizer connected.")
    print("PASS: Scheduler connected.")

    # TEST 4: Training epoch.

    print("\n[Test 4] Training Epoch")

    weights_before = [
        weight.copy()
        for weight in model.get_weights()
    ]

    train_loss = trainer.train_epoch(
        train_loader
    )

    assert isinstance(train_loss, float)
    assert tf.math.is_finite(train_loss)

    weights_after = model.get_weights()

    weights_updated = any(
        not (before == after).all()
        for before, after in zip(
            weights_before,
            weights_after,
        )
    )

    assert weights_updated

    assert trainer.global_step == len(train_loader)

    print(
        f"PASS: Training loss = {train_loss:.4f}"
    )

    print("PASS: Model weights were updated.")

    # TEST 5: Validation.

    print("\n[Test 5] Validation")

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    validation_weights_before = [
        weight.copy()
        for weight in model.get_weights()
    ]

    validation_results = evaluator.evaluate(
        val_loader
    )

    validation_weights_after = model.get_weights()

    assert tf.math.is_finite(
        validation_results["val_loss"]
    )

    assert validation_results["num_samples"] == 8
    assert validation_results["num_batches"] == 2

    weights_unchanged = all(
        (before == after).all()
        for before, after in zip(
            validation_weights_before,
            validation_weights_after,
        )
    )

    assert weights_unchanged

    print(
        f"PASS: Validation loss = "
        f"{validation_results['val_loss']:.4f}"
    )

    print("PASS: Validation did not update model weights.")

    # TEST 6: Scheduler callback.

    print("\n[Test 6] Scheduler Callback")

    scheduler.on_train_begin()

    initial_lr = trainer._get_learning_rate()

    scheduler.on_epoch_begin(0)

    updated_lr = trainer._get_learning_rate()

    assert initial_lr > 0
    assert updated_lr > 0

    scheduler.on_epoch_end(
        0,
        logs={
            "loss": train_loss,
            "val_loss": validation_results["val_loss"],
        },
    )

    scheduler.on_train_end()

    print("PASS: Scheduler callbacks executed.")
    print(
        f"Learning rate after scheduler: {updated_lr:.8f}"
    )

    # TEST 7: Loss output validation.

    print("\n[Test 7] Loss Validation")

    outputs = model(
        train_loader[0]["image"],
        train_loader[0]["captions"],
        training=False,
    )

    loss = loss_fn(
        outputs["image_embeddings"],
        outputs["text_embeddings"],
    )

    assert loss.shape.rank == 0
    assert tf.math.is_finite(loss)

    print("PASS: Loss is a finite scalar.")

    # Final results.

    print("\n" + "=" * 55)
    print("ALL TRAINING INTEGRATION TESTS PASSED!")
    print("=" * 55)

    print(
        f"Training Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{validation_results['val_loss']:.4f}"
    )

    print(
        "\nNote: Synthetic data was used. "
        "These are not real Flickr30k results."
    )


if __name__ == "__main__":
    test_training_integration()
