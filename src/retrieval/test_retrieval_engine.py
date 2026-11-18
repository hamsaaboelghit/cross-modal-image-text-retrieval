
import unittest

import tensorflow as tf

from src.retrieval.retrieval_engine import RetrievalEngine


class TestRetrievalEngine(unittest.TestCase):

    def setUp(self):
        self.image_embeddings = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )

        self.text_embeddings = tf.constant(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.1, 0.9, 0.0],
                ],
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.1, 0.9],
                ],
            ],
            dtype=tf.float32,
        )

        self.image_names = [
            "image_1.jpg",
            "image_2.jpg",
            "image_3.jpg",
        ]

        self.captions = [
            ["red object", "red item"],
            ["green object", "green item"],
            ["blue object", "blue item"],
        ]

        self.engine = RetrievalEngine(
            image_embeddings=self.image_embeddings,
            text_embeddings=self.text_embeddings,
            image_names=self.image_names,
            captions=self.captions,
        )

    def test_initialization(self):
        self.assertEqual(self.engine.num_images, 3)
        self.assertEqual(self.engine.captions_per_image, 2)
        self.assertEqual(self.engine.embedding_dim, 3)

    def test_similarity_matrix_shape(self):
        similarity = self.engine.compute_similarity_matrix()

        self.assertEqual(tuple(similarity.shape), (3, 6))

    def test_text_to_image(self):
        query = tf.constant([1.0, 0.0, 0.0])

        results = self.engine.text_to_image(query, top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 3)
        self.assertEqual(
            results[0][0]["result"],
            "image_1.jpg",
        )
        self.assertAlmostEqual(
            results[0][0]["score"],
            1.0,
            places=6,
        )

    def test_image_to_text(self):
        query = tf.constant([0.0, 1.0, 0.0])

        results = self.engine.image_to_text(query, top_k=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(
            results[0][0]["result"]["image_name"],
            "image_2.jpg",
        )
        self.assertEqual(
            results[0][0]["result"]["caption"],
            "green object",
        )

    def test_caption_mapping(self):
        query = tf.constant([0.0, 0.0, 1.0])

        results = self.engine.image_to_text(query, top_k=2)

        for result in results[0]:
            self.assertEqual(
                result["result"]["image_name"],
                "image_3.jpg",
            )

    def test_batch_queries(self):
        queries = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        results = self.engine.text_to_image(queries, top_k=1)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0][0]["result"],
            "image_1.jpg",
        )
        self.assertEqual(
            results[1][0]["result"],
            "image_2.jpg",
        )

    def test_top_k_larger_than_candidates(self):
        results = self.engine.text_to_image(
            tf.constant([1.0, 0.0, 0.0]),
            top_k=10,
        )

        self.assertEqual(len(results[0]), 3)

    def test_invalid_top_k(self):
        for invalid_k in (0, -1, True, 1.5):
            with self.subTest(top_k=invalid_k):
                with self.assertRaises((ValueError, TypeError)):
                    self.engine.text_to_image(
                        tf.constant([1.0, 0.0, 0.0]),
                        top_k=invalid_k,
                    )

    def test_invalid_query_rank(self):
        with self.assertRaises(ValueError):
            self.engine.text_to_image(
                tf.constant([[[1.0, 0.0, 0.0]]]),
            )

    def test_invalid_query_dimension(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.engine.text_to_image(
                tf.constant([1.0, 0.0, 0.0, 0.0]),
            )

    def test_invalid_embedding_shapes(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            RetrievalEngine(
                image_embeddings=tf.zeros((3, 4)),
                text_embeddings=tf.zeros((2, 5, 4)),
            )

    def test_invalid_image_names_length(self):
        with self.assertRaises(ValueError):
            RetrievalEngine(
                image_embeddings=self.image_embeddings,
                text_embeddings=self.text_embeddings,
                image_names=["only_one.jpg"],
            )

    def test_invalid_caption_count(self):
        with self.assertRaises(ValueError):
            RetrievalEngine(
                image_embeddings=self.image_embeddings,
                text_embeddings=self.text_embeddings,
                captions=[
                    ["one caption"],
                    ["caption one", "caption two"],
                    ["caption one", "caption two"],
                ],
            )

    def test_non_finite_embeddings(self):
        invalid_embeddings = tf.constant(
            [
                [float("nan"), 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )

        with self.assertRaises(tf.errors.InvalidArgumentError):
            RetrievalEngine(
                image_embeddings=invalid_embeddings,
                text_embeddings=self.text_embeddings,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
