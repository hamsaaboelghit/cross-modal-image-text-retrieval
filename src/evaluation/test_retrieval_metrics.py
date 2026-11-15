
import unittest

import torch

from src.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
)


class TestRetrievalMetrics(unittest.TestCase):

    def setUp(self):
        self.similarity = torch.tensor([
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.1, 0.9, 0.8, 0.7, 0.6],
            [0.1, 0.2, 0.3, 0.4, 0.9],
        ])

        self.relevance = torch.tensor([
            [1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
        ])

    def test_recall_metrics(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
            ks=(1, 2, 3, 5),
        )

        self.assertAlmostEqual(
            results["recall@1"], 2 / 3, places=6
        )

        self.assertAlmostEqual(
            results["recall@2"], 1.0, places=6
        )

        self.assertAlmostEqual(
            results["recall@3"], 1.0, places=6
        )

        self.assertAlmostEqual(
            results["recall@5"], 1.0, places=6
        )

    def test_mean_recall(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
            ks=(1, 2, 3),
        )

        self.assertAlmostEqual(
            results["mean_recall"], 8 / 9, places=6
        )

    def test_median_rank(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
        )

        self.assertEqual(results["median_rank"], 1.0)

    def test_mean_reciprocal_rank(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
        )

        self.assertAlmostEqual(
            results["mrr"], 5 / 6, places=6
        )

    def test_query_and_candidate_counts(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
        )

        self.assertEqual(results["num_queries"], 3.0)
        self.assertEqual(results["num_candidates"], 5.0)

    def test_k_larger_than_candidates(self):
        results = compute_retrieval_metrics(
            self.similarity,
            self.relevance,
            ks=(10,),
        )

        self.assertEqual(results["recall@10"], 1.0)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            compute_retrieval_metrics(
                torch.randn(3, 5),
                torch.randn(3, 4),
            )

    def test_query_without_positive(self):
        invalid_relevance = self.relevance.clone()
        invalid_relevance[0] = 0

        with self.assertRaises(ValueError):
            compute_retrieval_metrics(
                self.similarity,
                invalid_relevance,
            )

    def test_nan_scores(self):
        invalid_similarity = self.similarity.clone()
        invalid_similarity[0, 0] = float("nan")

        with self.assertRaises(ValueError):
            compute_retrieval_metrics(
                invalid_similarity,
                self.relevance,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
