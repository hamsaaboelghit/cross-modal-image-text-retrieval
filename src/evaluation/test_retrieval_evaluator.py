
import unittest

import torch
import torch.nn as nn

from src.evaluation.retrieval_evaluator import RetrievalEvaluator


class DummyRetrievalModel(nn.Module):
    """Small deterministic model used only for evaluator tests."""

    def forward(self, images, captions):
        image_embeddings = images[:, :2, 0, 0]
        text_embeddings = captions[:, :, :2].float()

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


class BatchNormPreservingEncoder(nn.Module):
    """Small encoder that keeps BatchNorm in evaluation mode."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def train(self, mode=True):
        super().train(mode)
        self.bn.eval()
        return self

    def forward(self, images):
        features = self.bn(images)
        return features.mean(dim=(2, 3))[:, :2]


class DummyModelWithBatchNorm(nn.Module):
    """Model used to test BatchNorm state preservation."""

    def __init__(self):
        super().__init__()
        self.image_encoder = BatchNormPreservingEncoder()

    def forward(self, images, captions):
        image_embeddings = self.image_encoder(images)
        text_embeddings = captions[:, :, :2].float()

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


class TestRetrievalEvaluator(unittest.TestCase):

    def setUp(self):
        self.model = DummyRetrievalModel()

        self.evaluator = RetrievalEvaluator(
            model=self.model,
            device="cpu",
            query_batch_size=2,
            candidate_chunk_size=2,
            ks=(1, 2, 5),
        )

    def test_perfect_retrieval_both_directions(self):
        image_embeddings = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        text_embeddings = torch.tensor([
            [[1.0, 0.0], [0.9, 0.1]],
            [[0.0, 1.0], [0.1, 0.9]],
        ])

        results = self.evaluator.evaluate_embeddings(
            image_embeddings,
            text_embeddings,
        )

        self.assertEqual(
            results["text_to_image"]["recall@1"],
            1.0,
        )

        self.assertEqual(
            results["image_to_text"]["recall@1"],
            1.0,
        )

        self.assertEqual(
            results["text_to_image"]["mrr"],
            1.0,
        )

        self.assertEqual(
            results["image_to_text"]["median_rank"],
            1.0,
        )

    def test_retrieval_ranks_with_candidate_chunking(self):
        queries = torch.tensor([
            [1.0, 0.0],
        ])

        candidates = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ])

        positive_indices = torch.tensor([
            [2],
        ])

        evaluator = RetrievalEvaluator(
            model=self.model,
            device="cpu",
            query_batch_size=1,
            candidate_chunk_size=1,
        )

        ranks = evaluator._compute_first_relevant_ranks(
            queries,
            candidates,
            positive_indices,
        )

        self.assertEqual(ranks.tolist(), [2])

    def test_stable_tie_breaking(self):
        queries = torch.tensor([
            [1.0, 0.0],
        ])

        candidates = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
        ])

        positive_indices = torch.tensor([
            [1],
        ])

        ranks = self.evaluator._compute_first_relevant_ranks(
            queries,
            candidates,
            positive_indices,
        )

        self.assertEqual(ranks.tolist(), [2])

    def test_metrics_calculation(self):
        ranks = torch.tensor([1, 2, 3])

        metrics = self.evaluator._metrics_from_ranks(
            ranks,
            num_candidates=5,
        )

        self.assertAlmostEqual(metrics["recall@1"], 1 / 3)
        self.assertAlmostEqual(metrics["recall@2"], 2 / 3)

        self.assertAlmostEqual(
            metrics["mean_recall"],
            (1 / 3 + 2 / 3 + 1.0) / 3,
        )

        self.assertAlmostEqual(
            metrics["mrr"],
            (1 + 1 / 2 + 1 / 3) / 3,
        )

    def test_extract_embeddings_from_dataloader(self):
        images = torch.tensor([
            [[[1.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
        ])

        captions = torch.tensor([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])

        dataloader = [
            {
                "image": images,
                "captions": captions,
                "image_name": ["image_1.jpg", "image_2.jpg"],
            }
        ]

        image_embeddings, text_embeddings, names = (
            self.evaluator.extract_embeddings(dataloader)
        )

        self.assertEqual(tuple(image_embeddings.shape), (2, 2))
        self.assertEqual(tuple(text_embeddings.shape), (2, 2, 2))
        self.assertEqual(names, ["image_1.jpg", "image_2.jpg"])

    def test_evaluate_dataloader(self):
        images = torch.tensor([
            [[[1.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
        ])

        captions = torch.tensor([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])

        dataloader = [
            {
                "image": images,
                "captions": captions,
                "image_name": ["image_1.jpg", "image_2.jpg"],
            }
        ]

        results = self.evaluator.evaluate(dataloader)

        self.assertIn("text_to_image", results)
        self.assertIn("image_to_text", results)

        self.assertEqual(results["num_images"], 2)
        self.assertEqual(results["captions_per_image"], 2)
        self.assertEqual(
            results["image_names"],
            ["image_1.jpg", "image_2.jpg"],
        )

    def test_batchnorm_state_preserved_after_evaluation(self):
        model = DummyModelWithBatchNorm()

        # The whole model is in training mode.
        model.train()

        # BatchNorm must remain in evaluation mode.
        self.assertTrue(model.training)
        self.assertFalse(model.image_encoder.bn.training)

        evaluator = RetrievalEvaluator(
            model=model,
            device="cpu",
            query_batch_size=2,
            candidate_chunk_size=2,
            ks=(1, 2),
        )

        images = torch.tensor([
            [[[1.0]], [[0.0]], [[0.0]]],
            [[[0.0]], [[1.0]], [[0.0]]],
        ])

        captions = torch.tensor([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])

        dataloader = [
            {
                "image": images,
                "captions": captions,
                "image_name": ["image_1.jpg", "image_2.jpg"],
            }
        ]

        evaluator.extract_embeddings(dataloader)

        # The original model state must be restored.
        self.assertTrue(model.training)

        # BatchNorm must still be in evaluation mode.
        self.assertFalse(model.image_encoder.bn.training)

    def test_model_eval_state_preserved_after_extraction(self):
        model = DummyModelWithBatchNorm()
        model.eval()

        evaluator = RetrievalEvaluator(
            model=model,
            device="cpu",
        )

        images = torch.randn(2, 3, 2, 2)
        captions = torch.tensor([
            [[1, 0]],
            [[0, 1]],
        ])

        dataloader = [
            {
                "image": images,
                "captions": captions,
                "image_name": ["image_1.jpg", "image_2.jpg"],
            }
        ]

        evaluator.extract_embeddings(dataloader)

        self.assertFalse(model.training)
        self.assertFalse(model.image_encoder.bn.training)

    def test_invalid_embedding_dimensions(self):
        image_embeddings = torch.randn(2, 3)
        text_embeddings = torch.randn(2, 2, 4)

        with self.assertRaises(ValueError):
            self.evaluator.evaluate_embeddings(
                image_embeddings,
                text_embeddings,
            )

    def test_empty_embeddings(self):
        image_embeddings = torch.empty(0, 2)
        text_embeddings = torch.empty(0, 5, 2)

        with self.assertRaises(ValueError):
            self.evaluator.evaluate_embeddings(
                image_embeddings,
                text_embeddings,
            )

    def test_duplicate_image_names(self):
        image_embeddings = torch.randn(2, 2)
        text_embeddings = torch.randn(2, 5, 2)

        with self.assertRaises(ValueError):
            self.evaluator.evaluate_embeddings(
                image_embeddings,
                text_embeddings,
                image_names=["same.jpg", "same.jpg"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
