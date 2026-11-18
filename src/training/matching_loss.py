
import tensorflow as tf


class MatchingBinaryCrossEntropyLoss(tf.keras.layers.Layer):
    """
    Binary classification loss for image-caption matching.

    Positive scores are logits for matching image-caption pairs.
    Negative scores are logits for mismatched image-caption pairs.

    Both inputs must have shape [B, C].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, positive_scores, negative_scores):
        positive_scores = tf.cast(
            tf.convert_to_tensor(positive_scores),
            tf.float32,
        )
        negative_scores = tf.cast(
            tf.convert_to_tensor(negative_scores),
            tf.float32,
        )

        tf.debugging.assert_rank(
            positive_scores,
            2,
            message="Positive scores must have shape [B, C].",
        )
        tf.debugging.assert_rank(
            negative_scores,
            2,
            message="Negative scores must have shape [B, C].",
        )

        tf.debugging.assert_equal(
            tf.shape(positive_scores),
            tf.shape(negative_scores),
            message="Positive and negative score shapes must match.",
        )

        tf.debugging.assert_positive(
            tf.shape(positive_scores)[0],
            message="The batch cannot be empty.",
        )
        tf.debugging.assert_positive(
            tf.shape(positive_scores)[1],
            message="At least one caption is required.",
        )

        tf.debugging.assert_all_finite(
            positive_scores,
            "Positive scores contain NaN or Inf.",
        )
        tf.debugging.assert_all_finite(
            negative_scores,
            "Negative scores contain NaN or Inf.",
        )

        positive_labels = tf.ones_like(positive_scores)
        negative_labels = tf.zeros_like(negative_scores)

        positive_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=positive_labels,
            logits=positive_scores,
        )

        negative_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=negative_labels,
            logits=negative_scores,
        )

        # Balance the positive and negative contributions.
        loss = (
            tf.reduce_mean(positive_loss)
            + tf.reduce_mean(negative_loss)
        ) / 2.0

        tf.debugging.assert_all_finite(
            loss,
            "Matching loss is not finite.",
        )

        return loss

    def get_config(self):
        return super().get_config()
