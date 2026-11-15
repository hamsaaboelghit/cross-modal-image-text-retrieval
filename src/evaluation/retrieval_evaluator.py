
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


class RetrievalEvaluator:
    """
    Memory-efficient bidirectional image-text retrieval evaluator.

    Supports:
        1. Text-to-Image retrieval.
        2. Image-to-Text retrieval.

    Uses chunked similarity computation to avoid creating
    extremely large similarity matrices.

    Expected model output:
        {
            "image_embeddings": [B, D],
            "text_embeddings": [B, C, D]
        }
    """

    def __init__(
        self,
        model,
        device: Optional[str] = None,
        query_batch_size: int = 32,
        candidate_chunk_size: int = 8192,
        ks: Iterable[int] = (1, 5, 10),
    ):

        if not isinstance(query_batch_size, int) or isinstance(
            query_batch_size, bool
        ) or query_batch_size <= 0:
            raise ValueError("query_batch_size must be a positive integer.")

        if not isinstance(candidate_chunk_size, int) or isinstance(
            candidate_chunk_size, bool
        ) or candidate_chunk_size <= 0:
            raise ValueError(
                "candidate_chunk_size must be a positive integer."
            )

        self.model = model

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        self.device = torch.device(
            device if device is not None else model_device
        )

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.query_batch_size = query_batch_size
        self.candidate_chunk_size = candidate_chunk_size

        self.ks = tuple(ks)

        if not self.ks:
            raise ValueError("At least one K value is required.")

        if any(
            not isinstance(k, int) or isinstance(k, bool) or k <= 0
            for k in self.ks
        ):
            raise ValueError("All K values must be positive integers.")

        if len(set(self.ks)) != len(self.ks):
            raise ValueError("K values must be unique.")

    @torch.inference_mode()
    def extract_embeddings(self, dataloader):
        """
        Extract embeddings from all batches.

        Returns:
            image_embeddings: [N, D], CPU tensor
            text_embeddings: [N, C, D], CPU tensor
            image_names: list[str]
        """

        was_training = self.model.training

        self.model.to(self.device)
        self.model.eval()

        all_image_embeddings = []
        all_text_embeddings = []
        all_image_names = []

        try:
            for batch in dataloader:

                if not isinstance(batch, dict):
                    raise TypeError(
                        "Each DataLoader batch must be a dictionary."
                    )

                required_keys = {"image", "captions", "image_name"}

                if not required_keys.issubset(batch):
                    raise KeyError(
                        f"Batch must contain: {required_keys}"
                    )

                images = batch["image"].to(
                    self.device,
                    non_blocking=True,
                )

                captions = batch["captions"].to(
                    self.device,
                    non_blocking=True,
                )

                if images.ndim != 4:
                    raise ValueError(
                        "Batch images must have shape [B, 3, H, W]."
                    )

                if captions.ndim != 3:
                    raise ValueError(
                        "Batch captions must have shape [B, C, L]."
                    )

                output = self.model(
                    images=images,
                    captions=captions,
                )

                if not isinstance(output, dict):
                    raise TypeError("Model output must be a dictionary.")

                if "image_embeddings" not in output:
                    raise KeyError(
                        "Model output is missing image_embeddings."
                    )

                if "text_embeddings" not in output:
                    raise KeyError(
                        "Model output is missing text_embeddings."
                    )

                image_embeddings = output["image_embeddings"]
                text_embeddings = output["text_embeddings"]

                if image_embeddings.ndim != 2:
                    raise ValueError(
                        "Image embeddings must have shape [B, D]."
                    )

                if text_embeddings.ndim != 3:
                    raise ValueError(
                        "Text embeddings must have shape [B, C, D]."
                    )

                batch_size = images.size(0)

                if image_embeddings.size(0) != batch_size:
                    raise ValueError(
                        "Image embedding batch size is incorrect."
                    )

                if text_embeddings.size(0) != batch_size:
                    raise ValueError(
                        "Text embedding batch size is incorrect."
                    )

                if text_embeddings.size(1) != captions.size(1):
                    raise ValueError(
                        "Number of text embeddings does not match "
                        "number of captions."
                    )

                if image_embeddings.size(1) != text_embeddings.size(2):
                    raise ValueError(
                        "Image and text embedding dimensions must match."
                    )

                if not torch.isfinite(image_embeddings).all():
                    raise ValueError(
                        "Image embeddings contain invalid values."
                    )

                if not torch.isfinite(text_embeddings).all():
                    raise ValueError(
                        "Text embeddings contain invalid values."
                    )

                names = batch["image_name"]

                if isinstance(names, str):
                    names = [names]
                else:
                    names = list(names)

                if len(names) != batch_size:
                    raise ValueError(
                        "Number of image names does not match batch size."
                    )

                all_image_embeddings.append(
                    image_embeddings.detach().float().cpu()
                )

                all_text_embeddings.append(
                    text_embeddings.detach().float().cpu()
                )

                all_image_names.extend(names)

        finally:
            self.model.train(was_training)

        if not all_image_embeddings:
            raise ValueError("DataLoader produced no batches.")

        image_embeddings = torch.cat(
            all_image_embeddings,
            dim=0,
        )

        text_embeddings = torch.cat(
            all_text_embeddings,
            dim=0,
        )

        if len(set(all_image_names)) != len(all_image_names):
            raise ValueError(
                "Duplicate image names detected in evaluation data."
            )

        return (
            F.normalize(image_embeddings, p=2, dim=-1),
            F.normalize(text_embeddings, p=2, dim=-1),
            all_image_names,
        )

    @torch.inference_mode()
    def _compute_first_relevant_ranks(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        positive_indices: torch.Tensor,
    ):
        """
        Compute the exact rank of the first relevant candidate.

        Similarities are calculated in chunks over candidates.

        Ties follow stable descending sorting:
        candidates with equal scores are ordered by their original index.

        Args:
            query_embeddings: [Q, D], CPU tensor.
            candidate_embeddings: [M, D], CPU tensor.
            positive_indices: [Q, P], CPU long tensor.

        Returns:
            first relevant ranks: [Q], CPU long tensor.
        """

        if query_embeddings.ndim != 2:
            raise ValueError("Query embeddings must be 2D.")

        if candidate_embeddings.ndim != 2:
            raise ValueError("Candidate embeddings must be 2D.")

        if positive_indices.ndim != 2:
            raise ValueError("Positive indices must be 2D.")

        if query_embeddings.size(0) != positive_indices.size(0):
            raise ValueError(
                "Query count and positive-index count must match."
            )

        if query_embeddings.size(1) != candidate_embeddings.size(1):
            raise ValueError(
                "Query and candidate dimensions must match."
            )

        num_candidates = candidate_embeddings.size(0)

        if num_candidates == 0:
            raise ValueError("Candidate gallery cannot be empty.")

        if positive_indices.size(1) == 0:
            raise ValueError("Every query must have a positive candidate.")

        if (
            positive_indices.min().item() < 0
            or positive_indices.max().item() >= num_candidates
        ):
            raise ValueError("Positive candidate index is out of range.")

        query_embeddings = F.normalize(
            query_embeddings.float(),
            p=2,
            dim=-1,
        )

        candidate_embeddings = F.normalize(
            candidate_embeddings.float(),
            p=2,
            dim=-1,
        )

        all_ranks = []

        for query_start in range(
            0,
            query_embeddings.size(0),
            self.query_batch_size,
        ):

            query_end = min(
                query_start + self.query_batch_size,
                query_embeddings.size(0),
            )

            queries = query_embeddings[query_start:query_end].to(
                self.device
            )

            positive_batch = positive_indices[
                query_start:query_end
            ]

            # Calculate similarities with all positive candidates.
            positive_embeddings = candidate_embeddings[
                positive_batch
            ].to(self.device)

            positive_scores = (
                queries.unsqueeze(1) * positive_embeddings
            ).sum(dim=-1)

            # Select the highest-scoring positive.
            # argmax selects the first positive in case of a tie.
            best_positive_scores, best_positive_positions = (
                positive_scores.max(dim=1)
            )

            best_positive_indices = positive_batch.to(
                self.device
            ).gather(
                1,
                best_positive_positions.unsqueeze(1),
            ).squeeze(1)

            # Number of candidates ranked before the best positive.
            ranks = torch.ones(
                queries.size(0),
                dtype=torch.long,
                device=self.device,
            )

            for candidate_start in range(
                0,
                num_candidates,
                self.candidate_chunk_size,
            ):

                candidate_end = min(
                    candidate_start + self.candidate_chunk_size,
                    num_candidates,
                )

                candidates = candidate_embeddings[
                    candidate_start:candidate_end
                ].to(self.device)

                similarities = queries @ candidates.T

                # Higher similarity always ranks first.
                greater_scores = (
                    similarities > best_positive_scores.unsqueeze(1)
                )

                # Equal scores rank first only if their original
                # candidate index is smaller.
                candidate_indices = torch.arange(
                    candidate_start,
                    candidate_end,
                    device=self.device,
                )

                equal_scores_before = (
                    (similarities == best_positive_scores.unsqueeze(1))
                    &
                    (
                        candidate_indices.unsqueeze(0)
                        < best_positive_indices.unsqueeze(1)
                    )
                )

                ranks += (
                    greater_scores | equal_scores_before
                ).sum(dim=1)

            all_ranks.append(ranks.cpu())

        return torch.cat(all_ranks, dim=0)

    def _metrics_from_ranks(self, ranks, num_candidates):
        """
        Calculate retrieval metrics from first-positive ranks.
        """

        if ranks.ndim != 1 or ranks.numel() == 0:
            raise ValueError("Ranks must be a non-empty 1D tensor.")

        ranks = ranks.float()

        results = {}
        recall_values = []

        for k in self.ks:

            effective_k = min(k, num_candidates)

            recall = (
                (ranks <= effective_k).float().mean().item()
            )

            results[f"recall@{k}"] = recall
            recall_values.append(recall)

        results["mean_recall"] = (
            sum(recall_values) / len(recall_values)
        )

        results["median_rank"] = ranks.median().item()

        results["mrr"] = (1.0 / ranks).mean().item()

        results["num_queries"] = float(ranks.numel())
        results["num_candidates"] = float(num_candidates)

        return results

    @torch.inference_mode()
    def evaluate_embeddings(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_names=None,
    ):
        """
        Evaluate already extracted embeddings in both directions.

        Args:
            image_embeddings: [N, D]
            text_embeddings: [N, C, D]

        Returns:
            Dictionary with Text-to-Image and Image-to-Text metrics.
        """

        if image_embeddings.ndim != 2:
            raise ValueError("Image embeddings must be 2D.")

        if text_embeddings.ndim != 3:
            raise ValueError("Text embeddings must be 3D.")

        num_images = image_embeddings.size(0)
        num_captions = text_embeddings.size(1)

        if num_images == 0 or num_captions == 0:
            raise ValueError("Evaluation embeddings cannot be empty.")

        if text_embeddings.size(0) != num_images:
            raise ValueError(
                "Image and text embedding counts must match."
            )

        if image_embeddings.size(1) != text_embeddings.size(2):
            raise ValueError(
                "Image and text embedding dimensions must match."
            )

        if image_names is not None:
            if len(image_names) != num_images:
                raise ValueError(
                    "image_names length must match image count."
                )

            if len(set(image_names)) != len(image_names):
                raise ValueError("Duplicate image names detected.")

        image_embeddings = F.normalize(
            image_embeddings.detach().float().cpu(),
            p=2,
            dim=-1,
        )

        text_embeddings = F.normalize(
            text_embeddings.detach().float().cpu(),
            p=2,
            dim=-1,
        )

        flat_text_embeddings = text_embeddings.reshape(
            num_images * num_captions,
            -1,
        )

        # Text-to-Image:
        # Each caption is relevant to its source image.
        text_queries = flat_text_embeddings

        text_positive_indices = torch.arange(
            num_images,
            dtype=torch.long,
        ).repeat_interleave(num_captions).unsqueeze(1)

        text_to_image_ranks = self._compute_first_relevant_ranks(
            query_embeddings=text_queries,
            candidate_embeddings=image_embeddings,
            positive_indices=text_positive_indices,
        )

        text_to_image_metrics = self._metrics_from_ranks(
            text_to_image_ranks,
            num_images,
        )

        # Image-to-Text:
        # Each image is relevant to all its own captions.
        image_queries = image_embeddings

        image_positive_indices = (
            torch.arange(num_images, dtype=torch.long).unsqueeze(1)
            * num_captions
            +
            torch.arange(num_captions, dtype=torch.long).unsqueeze(0)
        )

        image_to_text_ranks = self._compute_first_relevant_ranks(
            query_embeddings=image_queries,
            candidate_embeddings=flat_text_embeddings,
            positive_indices=image_positive_indices,
        )

        image_to_text_metrics = self._metrics_from_ranks(
            image_to_text_ranks,
            num_images * num_captions,
        )

        return {
            "text_to_image": text_to_image_metrics,
            "image_to_text": image_to_text_metrics,
            "num_images": num_images,
            "captions_per_image": num_captions,
            "embedding_dim": image_embeddings.size(1),
        }

    def evaluate(self, dataloader):
        """
        Extract embeddings from a DataLoader and evaluate retrieval.
        """

        (
            image_embeddings,
            text_embeddings,
            image_names,
        ) = self.extract_embeddings(dataloader)

        results = self.evaluate_embeddings(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            image_names=image_names,
        )

        results["image_names"] = image_names

        return results
