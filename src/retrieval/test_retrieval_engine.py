
import unittest

import torch

from src.retrieval.retrieval_engine import RetrievalEngine


class TestRetrievalEngine(unittest.TestCase):

    def setUp(self):

        # Three simple orthogonal image embeddings.
        self.image_embeddings = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        # Each image has two captions.
        self.text_embeddings = torch.tensor([
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
        ])

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

        self.assertEqual(similarity.shape, (3, 6))

    def test_text_to_image(self):

        query = torch.tensor([1.0, 0.0, 0.0])

        results = self.engine.text_to_image(
            query,
            top_k=3,
        )

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

        query = torch.tensor([0.0, 1.0, 0.0])

        results = self.engine.image_to_text(
            query,
            top_k=2,
        )

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

        results = self.engine.image_to_text(
            torch.tensor([0.0, 0.0, 1.0]),
            top_k=2,
        )

        for result in results[0]:

            self.assertEqual(
                result["result"]["image_name"],
                "image_3.jpg",
            )

    def test_batch_queries(self):

        queries = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        results = self.engine.text_to_image(
            queries,
            top_k=1,
        )

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
            torch.tensor([1.0, 0.0, 0.0]),
            top_k=10,
        )

        self.assertEqual(len(results[0]), 3)

    def test_invalid_top_k(self):

        with self.assertRaises(ValueError):
            self.engine.text_to_image(
                torch.tensor([1.0, 0.0, 0.0]),
                top_k=0,
            )

    def test_invalid_query_dimension(self):

        with self.assertRaises(ValueError):
            self.engine.text_to_image(
                torch.randn(5),
            )

    def test_invalid_embedding_shapes(self):

        with self.assertRaises(ValueError):

            RetrievalEngine(
                image_embeddings=torch.randn(3, 4),
                text_embeddings=torch.randn(2, 5, 4),
            )

    def test_invalid_image_names_length(self):

        with self.assertRaises(ValueError):

            RetrievalEngine(
                image_embeddings=self.image_embeddings,
                text_embeddings=self.text_embeddings,
                image_names=["only_one.jpg"],
            )

    def test_non_finite_embeddings(self):

        invalid_embeddings = self.image_embeddings.clone()
        invalid_embeddings[0, 0] = float("nan")

        with self.assertRaises(ValueError):

            RetrievalEngine(
                image_embeddings=invalid_embeddings,
                text_embeddings=self.text_embeddings,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
