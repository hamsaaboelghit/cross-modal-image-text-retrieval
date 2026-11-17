
import unittest

import tensorflow as tf

from src.evaluation.retrieval_evaluator import RetrievalEvaluator


class DummyRetrievalModel(tf.keras.Model):
    """Lightweight deterministic model for evaluator tests."""

    def call(self, images, captions, training=None):
        image_embeddings = tf.reduce_mean(
            images,
            axis=(1, 2),
        )[:, :2]

        flat_captions = tf.reshape(
            captions,
            (-1, tf.shape(captions)[-1]),
        )

        flat_text_embeddings = tf.cast(
            flat_captions[:, :2],
            tf.float32,
        )

        text_embeddings = tf.reshape(
            flat_text_embeddings,
            (
                tf.shape(captions)[0],
                tf.shape(captions)[1],
                2,
            ),
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


class TestRetrievalEvaluator(unittest.TestCase):

    def setUp(self):
        self.model = DummyRetrievalModel()

        self.evaluator = RetrievalEvaluator(
            model=self.model,
            query_batch_size=2,
            candidate_chunk_size=2,
            ks=(1, 2, 5),
        )

    def create_dataset(self):
        images = tf.concat(
            [
                tf.ones((1, 224, 224, 3))
                * tf.constant([1.0, 0.0, 0.0]),

                tf.ones((1, 224, 224, 3))
                * tf.constant([0.0, 1.0, 0.0]),
            ],
            axis=0,
        )

        captions = tf.constant(
            [
                [
                    [1, 0] + [0] * 32,
                    [1, 0] + [0] * 32,
                    [1, 0] + [0] * 32,
                    [1, 0] + [0] * 32,
                    [1, 0] + [0] * 32,
                ],
                [
                    [0, 1] + [0] * 32,
                    [0, 1] + [0] * 32,
                    [0, 1] + [0] * 32,
                    [0, 1] + [0] * 32,
                    [0, 1] + [0] * 32,
                ],
            ],
            dtype=tf.int32,
        )

        image_names = tf.constant(
            ["image_1.jpg", "image_2.jpg"]
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "image": images,
                "captions": captions,
                "image_name": image_names,
            }
        )

        return dataset.batch(2)

    def test_evaluator_initialization(self):
        self.assertIsNotNone(self.evaluator)
        self.assertIs(
            self.evaluator.model,
            self.model,
        )

    def test_extract_embeddings(self):
        dataset = self.create_dataset()

        (
            image_embeddings,
            text_embeddings,
            image_names,
        ) = self.evaluator.extract_embeddings(dataset)

        self.assertEqual(
            tuple(image_embeddings.shape),
            (2, 2),
        )

        self.assertEqual(
            tuple(text_embeddings.shape),
            (2, 5, 2),
        )

        self.assertEqual(
            image_names,
            ["image_1.jpg", "image_2.jpg"],
        )

    def test_evaluate_both_retrieval_directions(self):
        dataset = self.create_dataset()

        results = self.evaluator.evaluate(dataset)

        self.assertIn("text_to_image", results)
        self.assertIn("image_to_text", results)

        self.assertIn(
            "recall@1",
            results["text_to_image"],
        )

        self.assertIn(
            "recall@1",
            results["image_to_text"],
        )

        self.assertIn(
            "mrr",
            results["text_to_image"],
        )

        self.assertIn(
            "median_rank",
            results["image_to_text"],
        )

    def test_perfect_retrieval(self):
        dataset = self.create_dataset()

        results = self.evaluator.evaluate(dataset)

        self.assertAlmostEqual(
            results["text_to_image"]["recall@1"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            results["image_to_text"]["recall@1"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            results["text_to_image"]["mrr"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            results["image_to_text"]["mrr"],
            1.0,
            places=5,
        )

    def test_model_is_not_updated_during_evaluation(self):
        dataset = self.create_dataset()

        original_weights = [
            weight.numpy().copy()
            for weight in self.model.weights
        ]

        self.evaluator.evaluate(dataset)

        self.assertEqual(
            len(self.model.weights),
            len(original_weights),
        )

        for original, current in zip(
            original_weights,
            self.model.weights,
        ):
            tf.debugging.assert_near(
                original,
                current,
            )

    def test_invalid_constructor_arguments(self):
        with self.assertRaises((ValueError, TypeError)):
            RetrievalEvaluator(
                model=self.model,
                query_batch_size=0,
            )

        with self.assertRaises((ValueError, TypeError)):
            RetrievalEvaluator(
                model=self.model,
                candidate_chunk_size=0,
            )

        with self.assertRaises((ValueError, TypeError)):
            RetrievalEvaluator(
                model=self.model,
                ks=(),
            )

    def test_duplicate_image_names_are_rejected(self):
        dataset = self.create_dataset()

        duplicate_dataset = tf.data.Dataset.from_tensor_slices(
            {
                "image": tf.concat(
                    [
                        tf.ones((1, 224, 224, 3))
                        * tf.constant([1.0, 0.0, 0.0]),
                        tf.ones((1, 224, 224, 3))
                        * tf.constant([0.0, 1.0, 0.0]),
                    ],
                    axis=0,
                ),
                "captions": tf.constant(
                    [
                        [[1, 0] + [0] * 32] * 5,
                        [[0, 1] + [0] * 32] * 5,
                    ],
                    dtype=tf.int32,
                ),
                "image_name": tf.constant(
                    ["same.jpg", "same.jpg"]
                ),
            }
        ).batch(2)

        with self.assertRaises(ValueError):
            self.evaluator.extract_embeddings(
                duplicate_dataset
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
