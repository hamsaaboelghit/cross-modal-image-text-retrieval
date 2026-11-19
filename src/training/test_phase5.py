"""Focused tests for Phase 5 data and backbone controls."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.image_encoder import ImageEncoder
from src.training.train_config import TrainingConfig
from src.training.triplet_loss import TripletContrastiveLoss


class Phase5Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_dir = self.root / "images"
        self.encoded_dir = self.root / "encoded"
        self.image_dir.mkdir()
        fold_dir = self.encoded_dir / "fold_1"
        fold_dir.mkdir(parents=True)
        rows = []
        for i in range(4):
            name = f"image_{i}.jpg"
            Image.fromarray(np.full((240, 240, 3), 50 + i * 20, np.uint8)).save(
                self.image_dir / name
            )
            for caption_index in range(5):
                sequence = [2, 4 + caption_index, 3] + [0] * 31
                rows.append({
                    "image_name": name,
                    "caption_number": caption_index,
                    "dataset_split": "train",
                    "sequence": json.dumps(sequence),
                })
        pd.DataFrame(rows).to_csv(
            fold_dir / "encoded_captions.csv", index=False
        )

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, **kwargs):
        return Flickr30kDataset(
            1, "train", image_dir=self.image_dir,
            encoded_dir=self.encoded_dir, **kwargs
        )

    def test_one_caption_is_seeded_and_shape_is_dynamic(self):
        a = self.dataset(caption_count=1, caption_seed=77)
        b = self.dataset(caption_count=1, caption_seed=77)
        self.assertEqual(a[0]["captions"].shape, (1, 34))
        np.testing.assert_array_equal(a[0]["captions"], b[0]["captions"])

    def test_default_retains_five_captions(self):
        data = self.dataset()
        self.assertEqual(data[0]["captions"].shape, (5, 34))
        batch = next(iter(data.as_dataset(2, prefetch=False)))
        self.assertEqual(tuple(batch["captions"].shape), (2, 5, 34))

    def test_light_augmentation_is_valid_and_training_only(self):
        data = self.dataset(image_augmentation="light")
        batch = next(iter(data.as_dataset(2, prefetch=False)))
        self.assertEqual(tuple(batch["image"].shape), (2, 224, 224, 3))
        self.assertTrue(bool(tf.reduce_all(batch["image"] >= 0).numpy()))
        self.assertTrue(bool(tf.reduce_all(batch["image"] <= 255).numpy()))

    def test_caption_ablation_rejected_for_validation(self):
        with self.assertRaises(ValueError):
            Flickr30kDataset(
                1, "validation", image_dir=self.image_dir,
                encoded_dir=self.encoded_dir, caption_count=1
            )

    def test_augmentation_rejected_for_validation(self):
        with self.assertRaises(ValueError):
            Flickr30kDataset(
                1, "validation", image_dir=self.image_dir,
                encoded_dir=self.encoded_dir, image_augmentation="light"
            )

    def test_invalid_caption_count_rejected(self):
        with self.assertRaises(ValueError):
            self.dataset(caption_count=0)


class Phase5BackboneAndNegativeTests(unittest.TestCase):
    def test_train_all_backbone_freezes_batch_norm_only(self):
        encoder = ImageEncoder(pretrained=False, train_all_backbone=True)
        trainable = [
            layer for layer in encoder.backbone.layers
            if layer.trainable
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        batch_norm = [
            layer for layer in encoder.backbone.layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        self.assertGreater(len(trainable), 0)
        self.assertTrue(all(not layer.trainable for layer in batch_norm))

    def test_conflicting_backbone_modes_rejected(self):
        with self.assertRaises(ValueError):
            ImageEncoder(
                pretrained=False, unfreeze_last_block=True,
                train_all_backbone=True
            )

    def test_hardest_triplet_has_finite_gradients_and_deterministic_eval(self):
        tf.keras.utils.set_random_seed(42)
        images = tf.Variable(tf.random.normal((4, 16)))
        captions = tf.Variable(tf.random.normal((4, 3, 16)))
        loss_fn = TripletContrastiveLoss(
            margin=0.2, negative_sampling_strategy="hardest"
        )
        with tf.GradientTape() as tape:
            loss = loss_fn(images, captions, training=True)
        grads = tape.gradient(loss, [images, captions])
        self.assertTrue(bool(tf.math.is_finite(loss).numpy()))
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(
            bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in grads
        ))
        first = loss_fn(images, captions, training=False)
        second = loss_fn(images, captions, training=False)
        self.assertAlmostEqual(float(first.numpy()), float(second.numpy()), 7)

    def test_step_budget_configuration(self):
        config = TrainingConfig(
            max_train_steps_per_epoch=250,
            max_validation_steps=100,
        )
        self.assertEqual(config.max_train_steps_per_epoch, 250)
        self.assertEqual(config.max_validation_steps, 100)

        for field in ("max_train_steps_per_epoch", "max_validation_steps"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    TrainingConfig(**{field: 0})

    def test_invalid_negative_strategy_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(negative_sampling_strategy="unknown")
        with self.assertRaises(ValueError):
            TripletContrastiveLoss(negative_sampling_strategy="unknown")


if __name__ == "__main__":
    unittest.main()
