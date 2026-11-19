"""Unit tests for the three image-text retrieval objectives."""

import unittest
import tensorflow as tf

from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.margin_loss import MarginBasedContrastiveLoss
from src.training.triplet_loss import TripletContrastiveLoss


class LossComparisonTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(42)
        self.images = tf.Variable(tf.random.normal((4, 16)))
        self.captions = tf.Variable(tf.random.normal((4, 3, 16)))

    def assert_loss_and_gradients(self, criterion, **kwargs):
        with tf.GradientTape() as tape:
            loss = criterion(self.images, self.captions, **kwargs)
        self.assertEqual(loss.shape.rank, 0)
        self.assertTrue(bool(tf.math.is_finite(loss).numpy()))
        gradients = tape.gradient(loss, [self.images, self.captions])
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy())
                             for g in gradients))

    def test_infonce_training_interface_and_gradients(self):
        self.assert_loss_and_gradients(
            MultiPositiveInfoNCELoss(temperature=0.07), training=True
        )

    def test_margin_training_interface_and_gradients(self):
        self.assert_loss_and_gradients(
            MarginBasedContrastiveLoss(margin=1.0), training=True
        )

    def test_triplet_training_interface_and_gradients(self):
        self.assert_loss_and_gradients(
            TripletContrastiveLoss(margin=0.2), training=True
        )

    def test_triplet_validation_is_deterministic(self):
        criterion = TripletContrastiveLoss(margin=0.2)
        first = criterion(self.images, self.captions, training=False)
        second = criterion(self.images, self.captions, training=False)
        self.assertAlmostEqual(float(first.numpy()), float(second.numpy()), places=7)

    def test_all_losses_are_non_negative(self):
        criteria = [
            MultiPositiveInfoNCELoss(temperature=0.07),
            MarginBasedContrastiveLoss(margin=1.0),
            TripletContrastiveLoss(margin=0.2),
        ]
        for criterion in criteria:
            loss = criterion(self.images, self.captions, training=False)
            self.assertGreaterEqual(float(loss.numpy()), 0.0)


if __name__ == "__main__":
    unittest.main()
