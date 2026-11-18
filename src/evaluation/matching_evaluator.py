
import tensorflow as tf

from src.training.matching_loss import MatchingBinaryCrossEntropyLoss


class MatchingEvaluator:
    """
    Evaluate the cross-modal image-caption matching branch.

    Positive pairs:
        Original images with their own captions.

    Negative pairs:
        Cyclically shifted images with the original captions.

    The matching head returns logits:
        logit >= 0 -> positive
        logit < 0  -> negative
    """

    def __init__(self, model, threshold=0.0):
        if model is None:
            raise ValueError("model cannot be None.")

        if not getattr(model, "use_cross_modal_attention", False):
            raise ValueError(
                "MatchingEvaluator requires a model with "
                "use_cross_modal_attention=True."
            )

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be numeric.")

        self.model = model
        self.threshold = float(threshold)
        self.loss_fn = MatchingBinaryCrossEntropyLoss()

    @staticmethod
    def _prepare_batch(batch):
        if not isinstance(batch, dict):
            raise TypeError("Each batch must be a dictionary.")

        if "image" not in batch or "captions" not in batch:
            raise KeyError(
                "Batch must contain 'image' and 'captions'."
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

        tf.debugging.assert_equal(
            tf.shape(images)[0],
            tf.shape(captions)[0],
            message="Image and caption batch sizes must match.",
        )

        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must have shape [B,H,W,3].",
        )

        tf.debugging.assert_positive(
            tf.shape(images)[0],
            message="Batch cannot be empty.",
        )

        tf.debugging.assert_positive(
            tf.shape(captions)[1],
            message="At least one caption is required.",
        )

        return images, captions

    @staticmethod
    def _binary_metrics(labels, predictions):
        labels = tf.cast(labels, tf.int32)
        predictions = tf.cast(predictions, tf.int32)

        true_positive = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(labels, 1),
                    tf.equal(predictions, 1),
                ),
                tf.float32,
            )
        )

        true_negative = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(labels, 0),
                    tf.equal(predictions, 0),
                ),
                tf.float32,
            )
        )

        false_positive = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(labels, 0),
                    tf.equal(predictions, 1),
                ),
                tf.float32,
            )
        )

        false_negative = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(labels, 1),
                    tf.equal(predictions, 0),
                ),
                tf.float32,
            )
        )

        accuracy = (
            true_positive + true_negative
        ) / tf.maximum(
            true_positive
            + true_negative
            + false_positive
            + false_negative,
            1.0,
        )

        precision = true_positive / tf.maximum(
            true_positive + false_positive,
            1.0,
        )

        recall = true_positive / tf.maximum(
            true_positive + false_negative,
            1.0,
        )

        f1 = (
            2.0 * precision * recall
            / tf.maximum(precision + recall, 1e-8)
        )

        return {
            "accuracy": float(accuracy.numpy()),
            "precision": float(precision.numpy()),
            "recall": float(recall.numpy()),
            "f1_score": float(f1.numpy()),
            "true_positive": int(true_positive.numpy()),
            "true_negative": int(true_negative.numpy()),
            "false_positive": int(false_positive.numpy()),
            "false_negative": int(false_negative.numpy()),
        }

    def evaluate(self, dataset):
        """
        Evaluate matching performance over a dataset.

        Batches with fewer than two images are skipped because
        they cannot provide an in-batch negative pair.
        """

        all_positive_scores = []
        all_negative_scores = []

        skipped_batches = 0
        evaluated_batches = 0

        for batch in dataset:
            images, captions = self._prepare_batch(batch)

            batch_size = int(
                tf.shape(images)[0].numpy()
            )

            if batch_size < 2:
                skipped_batches += 1
                continue

            positive_outputs = self.model(
                images,
                captions,
                training=False,
            )

            negative_images = tf.roll(
                images,
                shift=1,
                axis=0,
            )

            negative_outputs = self.model(
                negative_images,
                captions,
                training=False,
            )

            if "matching_scores" not in positive_outputs:
                raise KeyError(
                    "Model output is missing matching_scores."
                )

            if "matching_scores" not in negative_outputs:
                raise KeyError(
                    "Negative model output is missing matching_scores."
                )

            positive_scores = tf.cast(
                positive_outputs["matching_scores"],
                tf.float32,
            )

            negative_scores = tf.cast(
                negative_outputs["matching_scores"],
                tf.float32,
            )

            tf.debugging.assert_rank(
                positive_scores,
                2,
                message="Positive scores must have shape [B,C].",
            )

            tf.debugging.assert_rank(
                negative_scores,
                2,
                message="Negative scores must have shape [B,C].",
            )

            tf.debugging.assert_equal(
                tf.shape(positive_scores),
                tf.shape(negative_scores),
                message="Positive and negative score shapes must match.",
            )

            tf.debugging.assert_all_finite(
                positive_scores,
                "Positive scores contain NaN or Inf.",
            )

            tf.debugging.assert_all_finite(
                negative_scores,
                "Negative scores contain NaN or Inf.",
            )

            # Keep the original [B, C] tensors for the matching loss.
            all_positive_scores.append(positive_scores)
            all_negative_scores.append(negative_scores)

            evaluated_batches += 1

        if not all_positive_scores:
            raise RuntimeError(
                "No batches with at least two images were available "
                "for matching evaluation."
            )

        # Concatenate batches while preserving [B, C].
        positive_scores_2d = tf.concat(
            all_positive_scores,
            axis=0,
        )

        negative_scores_2d = tf.concat(
            all_negative_scores,
            axis=0,
        )

        # Calculate matching loss using the required [B, C] shape.
        matching_loss = self.loss_fn(
            positive_scores_2d,
            negative_scores_2d,
        )

        # Flatten only for binary classification metrics.
        positive_scores = tf.reshape(
            positive_scores_2d,
            (-1,),
        )

        negative_scores = tf.reshape(
            negative_scores_2d,
            (-1,),
        )

        labels = tf.concat(
            [
                tf.ones_like(
                    positive_scores,
                    dtype=tf.int32,
                ),
                tf.zeros_like(
                    negative_scores,
                    dtype=tf.int32,
                ),
            ],
            axis=0,
        )

        scores = tf.concat(
            [
                positive_scores,
                negative_scores,
            ],
            axis=0,
        )

        predictions = tf.cast(
            scores >= self.threshold,
            tf.int32,
        )

        metrics = self._binary_metrics(
            labels,
            predictions,
        )

        metrics.update({
            "matching_loss": float(matching_loss.numpy()),
            "positive_mean_score": float(
                tf.reduce_mean(positive_scores).numpy()
            ),
            "negative_mean_score": float(
                tf.reduce_mean(negative_scores).numpy()
            ),
            "num_positive_pairs": int(
                tf.size(positive_scores).numpy()
            ),
            "num_negative_pairs": int(
                tf.size(negative_scores).numpy()
            ),
            "evaluated_batches": evaluated_batches,
            "skipped_batches": skipped_batches,
            "threshold": self.threshold,
        })

        return metrics


if __name__ == "__main__":
    print("MatchingEvaluator module imported successfully.")
