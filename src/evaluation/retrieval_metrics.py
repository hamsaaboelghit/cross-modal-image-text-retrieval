
from typing import Dict, Iterable

import torch


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    relevance_matrix: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute retrieval metrics for a similarity matrix.

    Args:
        similarity_matrix:
            Tensor of shape [num_queries, num_candidates].
            Higher scores indicate more similar items.

        relevance_matrix:
            Boolean or binary tensor with the same shape.
            True indicates that a candidate is relevant to a query.
            A query may have multiple relevant candidates.

        ks:
            Recall cutoffs, e.g. (1, 5, 10).

    Returns:
        Dictionary containing Recall@K, Mean Recall,
        Median Rank, MRR, and number of queries.
    """

    # 1. Validate input tensors
    if not isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = torch.as_tensor(similarity_matrix)

    if not isinstance(relevance_matrix, torch.Tensor):
        relevance_matrix = torch.as_tensor(relevance_matrix)

    if similarity_matrix.ndim != 2:
        raise ValueError(
            "similarity_matrix must be a 2D tensor."
        )

    if relevance_matrix.ndim != 2:
        raise ValueError(
            "relevance_matrix must be a 2D tensor."
        )

    if similarity_matrix.shape != relevance_matrix.shape:
        raise ValueError(
            "Similarity and relevance matrices must have "
            "the same shape."
        )

    num_queries, num_candidates = similarity_matrix.shape

    if num_queries == 0 or num_candidates == 0:
        raise ValueError(
            "Similarity matrix cannot be empty."
        )

    # 2. Validate similarity scores
    similarity_matrix = similarity_matrix.detach().float()

    if not torch.isfinite(similarity_matrix).all():
        raise ValueError(
            "Similarity matrix contains NaN or infinite values."
        )

    # 3. Validate relevance information
    relevance_matrix = relevance_matrix.detach().bool()

    relevant_per_query = relevance_matrix.sum(dim=1)

    if (relevant_per_query == 0).any():
        raise ValueError(
            "Every query must have at least one relevant candidate."
        )

    # 4. Validate K values
    ks = tuple(ks)

    if not ks:
        raise ValueError(
            "At least one K value must be provided."
        )

    if any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0
        for k in ks
    ):
        raise ValueError(
            "All K values must be positive integers."
        )

    if len(set(ks)) != len(ks):
        raise ValueError(
            "K values must be unique."
        )

    # 5. Rank candidates for every query
    sorted_indices = torch.argsort(
        similarity_matrix,
        dim=1,
        descending=True,
        stable=True,
    )

    sorted_relevance = torch.gather(
        relevance_matrix,
        dim=1,
        index=sorted_indices,
    )

    # 6. Find the rank of the first relevant candidate
    positions = torch.arange(
        1,
        num_candidates + 1,
        device=similarity_matrix.device,
    ).unsqueeze(0)

    relevant_positions = torch.where(
        sorted_relevance,
        positions,
        num_candidates + 1,
    )

    first_relevant_ranks = relevant_positions.min(dim=1).values

    # 7. Calculate Recall@K
    results = {}

    recall_values = []

    for k in ks:
        effective_k = min(k, num_candidates)

        hits = (
            first_relevant_ranks <= effective_k
        ).float()

        recall_at_k = hits.mean().item()

        results[f"recall@{k}"] = recall_at_k
        recall_values.append(recall_at_k)

    # 8. Mean Recall
    results["mean_recall"] = (
        sum(recall_values) / len(recall_values)
    )

    # 9. Median Rank
    results["median_rank"] = (
        first_relevant_ranks.float().median().item()
    )

    # 10. Mean Reciprocal Rank
    reciprocal_ranks = 1.0 / first_relevant_ranks.float()

    results["mrr"] = reciprocal_ranks.mean().item()

    # 11. Additional information
    results["num_queries"] = float(num_queries)
    results["num_candidates"] = float(num_candidates)

    return results
