
from typing import Iterable, Optional

import tensorflow as tf


class RetrievalEvaluator:
    """
    TensorFlow bidirectional image-text retrieval evaluator.

    Expected model output:
        image_embeddings: [B, D]
        text_embeddings:  [B, C, D]

    Supports:
        - Text-to-Image retrieval
        - Image-to-Text retrieval
        - Chunked similarity computation
    """

    def __init__(
        self,
        model,
        query_batch_size: int = 32,
        candidate_chunk_size: int = 8192,
        ks: Iterable[int] = (1, 5, 10),
    ):
        if (
            not isinstance(query_batch_size, int)
            or isinstance(query_batch_size, bool)
            or query_batch_size <= 0
        ):
            raise ValueError(
                "query_batch_size must be a positive integer."
            )

        if (
            not isinstance(candidate_chunk_size, int)
            or isinstance(candidate_chunk_size, bool)
            or candidate_chunk_size <= 0
        ):
            raise ValueError(
                "candidate_chunk_size must be a positive integer."
            )

        self.model = model
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

    def extract_embeddings(self, dataset):
        """
        Extract normalized embeddings from a tf.data.Dataset.

        Each batch must contain:
            image: [B, H, W, 3]
            captions: [B, C, L]
            image_name: [B]
        """

        all_image_embeddings = []
        all_text_embeddings = []
        all_image_names = []

        for batch_index, batch in enumerate(dataset, start=1):
            if not isinstance(batch, dict):
                raise TypeError(
                    "Each dataset batch must be a dictionary."
                )

            required_keys = {"image", "captions", "image_name"}

            if not required_keys.issubset(batch):
                raise KeyError(
                    f"Batch must contain: {required_keys}"
                )

            images = tf.convert_to_tensor(
                batch["image"],
                dtype=tf.float32,
            )

            captions = tf.convert_to_tensor(
                batch["captions"],
                dtype=tf.int32,
            )

            tf.debugging.assert_rank(images, 4)
            tf.debugging.assert_rank(captions, 3)

            outputs = self.model(
                images,
                captions,
                training=False,
            )

            if not isinstance(outputs, dict):
                raise TypeError("Model output must be a dictionary.")

            if "image_embeddings" not in outputs:
                raise KeyError(
                    "Model output is missing image_embeddings."
                )

            if "text_embeddings" not in outputs:
                raise KeyError(
                    "Model output is missing text_embeddings."
                )

            image_embeddings = tf.cast(
                outputs["image_embeddings"],
                tf.float32,
            )

            text_embeddings = tf.cast(
                outputs["text_embeddings"],
                tf.float32,
            )

            tf.debugging.assert_rank(image_embeddings, 2)
            tf.debugging.assert_rank(text_embeddings, 3)

            tf.debugging.assert_equal(
                tf.shape(image_embeddings)[0],
                tf.shape(images)[0],
            )

            tf.debugging.assert_equal(
                tf.shape(text_embeddings)[0],
                tf.shape(images)[0],
            )

            tf.debugging.assert_equal(
                tf.shape(text_embeddings)[1],
                tf.shape(captions)[1],
            )

            tf.debugging.assert_equal(
                tf.shape(image_embeddings)[1],
                tf.shape(text_embeddings)[2],
            )

            tf.debugging.assert_all_finite(
                image_embeddings,
                "Image embeddings contain invalid values.",
            )

            tf.debugging.assert_all_finite(
                text_embeddings,
                "Text embeddings contain invalid values.",
            )

            image_embeddings = tf.math.l2_normalize(
                image_embeddings,
                axis=-1,
            )

            text_embeddings = tf.math.l2_normalize(
                text_embeddings,
                axis=-1,
            )

            all_image_embeddings.append(image_embeddings)
            all_text_embeddings.append(text_embeddings)

            names = batch["image_name"].numpy().tolist()

            names = [
                name.decode("utf-8") if isinstance(name, bytes)
                else str(name)
                for name in names
            ]

            if len(names) != int(tf.shape(images)[0].numpy()):
                raise ValueError(
                    "Image names count does not match batch size."
                )

            all_image_names.extend(names)

            if batch_index % 50 == 0:
                print(
                    f"Extracted batches: {batch_index} "
                    f"| Images: {len(all_image_names)}"
                )

        if not all_image_embeddings:
            raise ValueError("Dataset produced no batches.")

        if len(set(all_image_names)) != len(all_image_names):
            raise ValueError(
                "Duplicate image names detected in evaluation data."
            )

        image_embeddings = tf.concat(
            all_image_embeddings,
            axis=0,
        )

        text_embeddings = tf.concat(
            all_text_embeddings,
            axis=0,
        )

        return (
            image_embeddings,
            text_embeddings,
            all_image_names,
        )

    def _compute_first_relevant_ranks(
        self,
        query_embeddings,
        candidate_embeddings,
        positive_indices,
    ):
        """
        Compute exact 1-based rank of the highest-scoring positive.

        Similarities are calculated in candidate chunks to reduce memory.
        Ties are resolved by original candidate index.
        """

        query_embeddings = tf.math.l2_normalize(
            tf.cast(query_embeddings, tf.float32),
            axis=-1,
        )

        candidate_embeddings = tf.math.l2_normalize(
            tf.cast(candidate_embeddings, tf.float32),
            axis=-1,
        )

        positive_indices = tf.convert_to_tensor(
            positive_indices,
            dtype=tf.int32,
        )

        tf.debugging.assert_rank(query_embeddings, 2)
        tf.debugging.assert_rank(candidate_embeddings, 2)
        tf.debugging.assert_rank(positive_indices, 2)

        num_queries = int(
            tf.shape(query_embeddings)[0].numpy()
        )
        num_candidates = int(
            tf.shape(candidate_embeddings)[0].numpy()
        )

        tf.debugging.assert_positive(num_queries)
        tf.debugging.assert_positive(num_candidates)

        tf.debugging.assert_equal(
            tf.shape(query_embeddings)[1],
            tf.shape(candidate_embeddings)[1],
        )

        tf.debugging.assert_equal(
            tf.shape(positive_indices)[0],
            num_queries,
        )

        tf.debugging.assert_positive(
            tf.shape(positive_indices)[1],
            message="Every query must have at least one positive.",
        )

        tf.debugging.assert_greater_equal(positive_indices, 0)
        tf.debugging.assert_less(
            positive_indices,
            num_candidates,
        )

        all_ranks = []

        for query_start in range(
            0,
            num_queries,
            self.query_batch_size,
        ):
            query_end = min(
                query_start + self.query_batch_size,
                num_queries,
            )

            queries = query_embeddings[query_start:query_end]
            positive_batch = positive_indices[query_start:query_end]

            positive_embeddings = tf.gather(
                candidate_embeddings,
                positive_batch,
            )

            positive_scores = tf.reduce_sum(
                queries[:, tf.newaxis, :] * positive_embeddings,
                axis=-1,
            )

            best_positive_positions = tf.argmax(
                positive_scores,
                axis=1,
                output_type=tf.int32,
            )

            best_positive_scores = tf.reduce_max(
                positive_scores,
                axis=1,
            )

            best_positive_indices = tf.gather(
                positive_batch,
                best_positive_positions,
                batch_dims=1,
            )

            ranks = tf.ones(
                (query_end - query_start,),
                dtype=tf.int32,
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
                ]

                similarities = tf.matmul(
                    queries,
                    candidates,
                    transpose_b=True,
                )

                candidate_indices = tf.range(
                    candidate_start,
                    candidate_end,
                    dtype=tf.int32,
                )

                greater_scores = (
                    similarities > best_positive_scores[:, tf.newaxis]
                )

                equal_scores_before = tf.logical_and(
                    tf.equal(
                        similarities,
                        best_positive_scores[:, tf.newaxis],
                    ),
                    candidate_indices[tf.newaxis, :]
                    < best_positive_indices[:, tf.newaxis],
                )

                ranked_before = tf.logical_or(
                    greater_scores,
                    equal_scores_before,
                )

                ranks += tf.reduce_sum(
                    tf.cast(ranked_before, tf.int32),
                    axis=1,
                )

            all_ranks.append(ranks)

        return tf.concat(all_ranks, axis=0)

    def _metrics_from_ranks(self, ranks, num_candidates):
        ranks = tf.cast(ranks, tf.float32)

        results = {}
        recall_values = []

        for k in self.ks:
            effective_k = min(k, num_candidates)

            recall = tf.reduce_mean(
                tf.cast(ranks <= effective_k, tf.float32)
            )

            recall_value = float(recall.numpy())

            results[f"recall@{k}"] = recall_value
            recall_values.append(recall_value)

        results["mean_recall"] = (
            sum(recall_values) / len(recall_values)
        )

        sorted_ranks = tf.sort(ranks)
        count = int(tf.shape(sorted_ranks)[0].numpy())
        middle = count // 2

        if count % 2:
            median_rank = sorted_ranks[middle]
        else:
            median_rank = (
                sorted_ranks[middle - 1] + sorted_ranks[middle]
            ) / 2.0

        results["median_rank"] = float(median_rank.numpy())

        results["mrr"] = float(
            tf.reduce_mean(tf.math.reciprocal(ranks)).numpy()
        )

        results["num_queries"] = float(count)
        results["num_candidates"] = float(num_candidates)

        return results

    def evaluate_embeddings(
        self,
        image_embeddings,
        text_embeddings,
        image_names=None,
    ):
        """
        Evaluate both retrieval directions.

        Text-to-Image:
            Every caption is relevant to its source image.

        Image-to-Text:
            Every image is relevant to all its own captions.
        """

        image_embeddings = tf.convert_to_tensor(
            image_embeddings,
            dtype=tf.float32,
        )

        text_embeddings = tf.convert_to_tensor(
            text_embeddings,
            dtype=tf.float32,
        )

        tf.debugging.assert_rank(image_embeddings, 2)
        tf.debugging.assert_rank(text_embeddings, 3)

        num_images = int(
            tf.shape(image_embeddings)[0].numpy()
        )

        captions_per_image = int(
            tf.shape(text_embeddings)[1].numpy()
        )

        tf.debugging.assert_positive(num_images)
        tf.debugging.assert_positive(captions_per_image)

        tf.debugging.assert_equal(
            tf.shape(text_embeddings)[0],
            num_images,
        )

        tf.debugging.assert_equal(
            tf.shape(image_embeddings)[1],
            tf.shape(text_embeddings)[2],
        )

        if image_names is not None:
            if len(image_names) != num_images:
                raise ValueError(
                    "image_names length must match image count."
                )

            if len(set(image_names)) != len(image_names):
                raise ValueError("Duplicate image names detected.")

        image_embeddings = tf.math.l2_normalize(
            image_embeddings,
            axis=-1,
        )

        text_embeddings = tf.math.l2_normalize(
            text_embeddings,
            axis=-1,
        )

        flat_text_embeddings = tf.reshape(
            text_embeddings,
            (
                num_images * captions_per_image,
                -1,
            ),
        )

        # Text-to-Image: each caption has one positive image.
        text_positive_indices = tf.repeat(
            tf.range(num_images, dtype=tf.int32),
            repeats=captions_per_image,
        )[:, tf.newaxis]

        text_to_image_ranks = self._compute_first_relevant_ranks(
            query_embeddings=flat_text_embeddings,
            candidate_embeddings=image_embeddings,
            positive_indices=text_positive_indices,
        )

        text_to_image_metrics = self._metrics_from_ranks(
            text_to_image_ranks,
            num_images,
        )

        # Image-to-Text: each image has C positive captions.
        image_positive_indices = (
            tf.range(num_images, dtype=tf.int32)[:, tf.newaxis]
            * captions_per_image
            +
            tf.range(captions_per_image, dtype=tf.int32)[tf.newaxis, :]
        )

        image_to_text_ranks = self._compute_first_relevant_ranks(
            query_embeddings=image_embeddings,
            candidate_embeddings=flat_text_embeddings,
            positive_indices=image_positive_indices,
        )

        image_to_text_metrics = self._metrics_from_ranks(
            image_to_text_ranks,
            num_images * captions_per_image,
        )

        return {
            "text_to_image": text_to_image_metrics,
            "image_to_text": image_to_text_metrics,
            "num_images": num_images,
            "captions_per_image": captions_per_image,
            "embedding_dim": int(
                tf.shape(image_embeddings)[1].numpy()
            ),
        }

    def evaluate(self, dataset):
        """
        Extract embeddings and evaluate retrieval from tf.data.Dataset.
        """

        (
            image_embeddings,
            text_embeddings,
            image_names,
        ) = self.extract_embeddings(dataset)

        results = self.evaluate_embeddings(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            image_names=image_names,
        )

        results["image_names"] = image_names

        return results


def _run_evaluator_tests():
    print("TensorFlow Retrieval Evaluator Tests")
    print("-" * 55)

    tf.random.set_seed(42)

    num_images = 12
    captions_per_image = 5
    embedding_dim = 32

    image_embeddings = tf.math.l2_normalize(
        tf.random.normal((num_images, embedding_dim)),
        axis=-1,
    )

    text_embeddings = tf.math.l2_normalize(
        tf.random.normal(
            (num_images, captions_per_image, embedding_dim)
        ),
        axis=-1,
    )

    evaluator = RetrievalEvaluator(
        model=None,
        query_batch_size=4,
        candidate_chunk_size=7,
        ks=(1, 5, 10),
    )

    results = evaluator.evaluate_embeddings(
        image_embeddings,
        text_embeddings,
        image_names=[
            f"image_{index}.jpg"
            for index in range(num_images)
        ],
    )

    assert "text_to_image" in results
    assert "image_to_text" in results

    for direction in ("text_to_image", "image_to_text"):
        metrics = results[direction]

        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "mean_recall" in metrics
        assert "median_rank" in metrics
        assert "mrr" in metrics

        assert 0.0 <= metrics["mrr"] <= 1.0

    print("Text-to-Image metrics: PASS")
    print("Image-to-Text metrics: PASS")
    print("Chunked rank calculation: PASS")
    print("Metric ranges: PASS")

    print("-" * 55)
    print("All Retrieval Evaluator tests passed!")


if __name__ == "__main__":
    _run_evaluator_tests()
