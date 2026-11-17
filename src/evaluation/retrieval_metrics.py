
from typing import Dict, Iterable

import tensorflow as tf


def compute_retrieval_metrics(
    similarity_matrix,
    relevance_matrix,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute retrieval metrics using TensorFlow.

    Args:
        similarity_matrix: [num_queries, num_candidates].
        relevance_matrix: Boolean/binary matrix with the same shape.
        ks: Recall cutoffs.

    Returns:
        Recall@K, Mean Recall, Median Rank, MRR,
        number of queries and candidates.
    """

    similarity_matrix = tf.convert_to_tensor(
        similarity_matrix,
        dtype=tf.float32,
    )
    relevance_matrix = tf.convert_to_tensor(relevance_matrix)

    tf.debugging.assert_rank(similarity_matrix, 2)
    tf.debugging.assert_rank(relevance_matrix, 2)
    tf.debugging.assert_equal(
        tf.shape(similarity_matrix),
        tf.shape(relevance_matrix),
    )
    tf.debugging.assert_positive(tf.shape(similarity_matrix))

    tf.debugging.assert_all_finite(
        similarity_matrix,
        "Similarity matrix contains NaN or infinite values.",
    )

    relevance_matrix = tf.cast(relevance_matrix, tf.bool)

    relevant_counts = tf.reduce_sum(
        tf.cast(relevance_matrix, tf.int32),
        axis=1,
    )

    tf.debugging.assert_positive(
        relevant_counts,
        message="Every query must have at least one relevant candidate.",
    )

    ks = tuple(ks)

    if not ks:
        raise ValueError("At least one K value must be provided.")

    if any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0
        for k in ks
    ):
        raise ValueError("All K values must be positive integers.")

    if len(set(ks)) != len(ks):
        raise ValueError("K values must be unique.")

    num_queries = int(tf.shape(similarity_matrix)[0].numpy())
    num_candidates = int(tf.shape(similarity_matrix)[1].numpy())

    sorted_indices = tf.argsort(
        similarity_matrix,
        axis=1,
        direction="DESCENDING",
        stable=True,
    )

    sorted_relevance = tf.gather(
        relevance_matrix,
        sorted_indices,
        axis=1,
        batch_dims=1,
    )

    positions = tf.range(
        1,
        num_candidates + 1,
        dtype=tf.int32,
    )[tf.newaxis, :]

    relevant_positions = tf.where(
        sorted_relevance,
        tf.broadcast_to(
            positions,
            tf.shape(sorted_relevance),
        ),
        num_candidates + 1,
    )

    first_relevant_ranks = tf.reduce_min(
        relevant_positions,
        axis=1,
    )

    results = {}
    recall_values = []

    for k in ks:
        effective_k = min(k, num_candidates)

        recall = tf.reduce_mean(
            tf.cast(
                first_relevant_ranks <= effective_k,
                tf.float32,
            )
        )

        recall_value = float(recall.numpy())

        results[f"recall@{k}"] = recall_value
        recall_values.append(recall_value)

    results["mean_recall"] = sum(recall_values) / len(recall_values)

    results["median_rank"] = float(
        tfp_median(first_relevant_ranks)
    )

    reciprocal_ranks = tf.math.reciprocal(
        tf.cast(first_relevant_ranks, tf.float32)
    )

    results["mrr"] = float(
        tf.reduce_mean(reciprocal_ranks).numpy()
    )

    results["num_queries"] = float(num_queries)
    results["num_candidates"] = float(num_candidates)

    return results


def tfp_median(values):
    """Calculate the statistical median without TensorFlow Probability."""

    sorted_values = tf.sort(values)
    count = tf.shape(sorted_values)[0]

    middle = count // 2

    if int(count.numpy()) % 2 == 1:
        return tf.cast(sorted_values[middle], tf.float32).numpy()

    lower = tf.cast(sorted_values[middle - 1], tf.float32)
    upper = tf.cast(sorted_values[middle], tf.float32)

    return float(((lower + upper) / 2.0).numpy())
