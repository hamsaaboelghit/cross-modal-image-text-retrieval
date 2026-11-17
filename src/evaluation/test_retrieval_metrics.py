
import unittest

import tensorflow as tf

from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
)


class TestRetrievalMetrics(unittest.TestCase):

    def setUp(self):
        self.similarity = tf.constant(
            [
                [0.9, 0.8, 0.7, 0.6, 0.5],
                [0.1, 0.9, 0.8, 0.7, 0.6],
                [0.1, 0.2, 0.3, 0.4, 0.9],
            ],
            dtype=tf.float32,
        )

        self.relevance = tf.constant(
            [
                [1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1],
            ],
            dtype=tf.int32,
        )

    def test_recall_at_k(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
            ks=(1, 2, 3, 5),
        )

        self.assertAlmostEqual(
            results["recall@1"],
            2 / 3,
            places=5,
        )

        self.assertAlmostEqual(
            results["recall@2"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            results["recall@3"],
            1.0,
            places=5,
        )

        self.assertAlmostEqual(
            results["recall@5"],
            1.0,
            places=5,
        )

    def test_mean_recall(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
            ks=(1, 2, 3),
        )

        expected = (
            2 / 3 + 1.0 + 1.0
        ) / 3

        self.assertAlmostEqual(
            results["mean_recall"],
            expected,
            places=5,
        )

    def test_median_rank(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
        )

        self.assertAlmostEqual(
            results["median_rank"],
            1.0,
            places=5,
        )

    def test_mean_reciprocal_rank(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
        )

        # First relevant ranks are 1, 2, and 1.
        expected = (
            1.0 + 0.5 + 1.0
        ) / 3

        self.assertAlmostEqual(
            results["mrr"],
            expected,
            places=5,
        )

    def test_query_and_candidate_counts(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
        )

        self.assertEqual(
            results["num_queries"],
            3,
        )

        self.assertEqual(
            results["num_candidates"],
            5,
        )

    def test_k_larger_than_candidates(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
            ks=(10,),
        )

        self.assertAlmostEqual(
            results["recall@10"],
            1.0,
            places=5,
        )

    def test_invalid_similarity_relevance_shape(self):
        invalid_relevance = tf.zeros(
            (3, 4),
            dtype=tf.int32,
        )

        with self.assertRaises(
            (
                ValueError,
                tf.errors.InvalidArgumentError,
            )
        ):
            compute_retrieval_metrics(
                self.similarity,
                invalid_relevance,
            )

    def test_query_without_positive_candidate(self):
        invalid_relevance = tf.tensor_scatter_nd_update(
            self.relevance,
            indices=[[0, 0], [0, 3]],
            updates=[0, 0],
        )

        with self.assertRaises(
            (
                ValueError,
                tf.errors.InvalidArgumentError,
            )
        ):
            compute_retrieval_metrics(
                self.similarity,
                invalid_relevance,
            )

    def test_nan_similarity_scores(self):
        invalid_similarity = tf.tensor_scatter_nd_update(
            self.similarity,
            indices=[[0, 0]],
            updates=[float("nan")],
        )

        with self.assertRaises(
            (
                ValueError,
                tf.errors.InvalidArgumentError,
            )
        ):
            compute_retrieval_metrics(
                invalid_similarity,
                self.relevance,
            )

    def test_empty_similarity_matrix(self):
        with self.assertRaises(
            (
                ValueError,
                tf.errors.InvalidArgumentError,
            )
        ):
            compute_retrieval_metrics(
                tf.zeros((0, 5)),
                tf.zeros((0, 5)),
            )

    def test_stable_ranking_for_equal_scores(self):
        similarity = tf.constant(
            [[0.9, 0.9, 0.5]],
            dtype=tf.float32,
        )

        relevance = tf.constant(
            [[0, 1, 0]],
            dtype=tf.int32,
        )

        results = compute_retrieval_metrics(
            similarity,
            relevance,
            ks=(1, 2),
        )

        # Stable sorting preserves candidate order.
        # The relevant candidate is ranked second.
        self.assertAlmostEqual(
            results["recall@1"],
            0.0,
            places=5,
        )

        self.assertAlmostEqual(
            results["recall@2"],
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
