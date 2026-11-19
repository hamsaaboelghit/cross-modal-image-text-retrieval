"""Tests for bounded training and validation loops."""
import unittest

import tensorflow as tf

from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


class TinyRetrievalModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.image_projection = tf.keras.layers.Dense(8)
        self.caption_embedding = self.add_weight(
            shape=(5, 8), initializer="ones", trainable=True
        )

    def call(self, images, captions, training=False):
        image_features = tf.reduce_mean(images, axis=[1, 2])
        image_embeddings = self.image_projection(image_features)
        batch_size = tf.shape(images)[0]
        text_embeddings = tf.tile(
            self.caption_embedding[None, :, :], [batch_size, 1, 1]
        )
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


def tiny_loss(image_embeddings, text_embeddings, training=False):
    differences = image_embeddings[:, None, :] - text_embeddings
    return tf.reduce_mean(tf.square(differences))


def make_batches(count):
    return [
        {
            "image": tf.ones((4, 224, 224, 3), dtype=tf.float32) * (i + 1),
            "captions": tf.zeros((4, 5, 34), dtype=tf.int32),
        }
        for i in range(count)
    ]


class CountingIterable:
    def __init__(self, items):
        self.items = items
        self.yielded = 0

    def __iter__(self):
        for item in self.items:
            self.yielded += 1
            yield item


class StepBudgetTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)
        self.config = TrainingConfig(
            epochs=1,
            batch_size=4,
            optimizer="adam",
            scheduler="none",
            device="cpu",
            max_train_steps_per_epoch=2,
            max_validation_steps=2,
        )

    def test_training_stops_at_configured_step_limit(self):
        model = TinyRetrievalModel()
        trainer = Trainer(model, tiny_loss, self.config)
        batches = CountingIterable(make_batches(5))

        trainer.train_epoch(
            batches,
            max_steps=self.config.max_train_steps_per_epoch,
        )

        self.assertEqual(batches.yielded, 2)
        self.assertEqual(trainer.global_step, 2)

    def test_validation_stops_at_configured_batch_limit(self):
        model = TinyRetrievalModel()
        model(
            tf.zeros((1, 224, 224, 3)),
            tf.zeros((1, 5, 34), dtype=tf.int32),
        )
        evaluator = Evaluator(model, tiny_loss, self.config)
        batches = CountingIterable(make_batches(5))

        result = evaluator.evaluate(
            batches,
            max_batches=self.config.max_validation_steps,
        )

        self.assertEqual(batches.yielded, 2)
        self.assertEqual(result["num_batches"], 2)
        self.assertEqual(result["num_samples"], 8)


if __name__ == "__main__":
    unittest.main()
