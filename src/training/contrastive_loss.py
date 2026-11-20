
import tensorflow as tf


class MultiPositiveInfoNCELoss(tf.keras.layers.Layer):
    """
    Symmetric multi-positive InfoNCE loss with optional
    cross-modal matching supervision.

    image_embeddings:
        [B, D]

    text_embeddings:
        [B, C, D]

    matching_scores:
        [B, C] positive image-caption matching logits.

    negative_matching_scores:
        [B, C] mismatched image-caption logits.
    """

    def __init__(
        self,
        temperature=0.07,
        matching_loss_weight=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        if matching_loss_weight < 0:
            raise ValueError(
                "matching_loss_weight cannot be negative."
            )

        self.temperature = float(temperature)
        self.matching_loss_weight = float(
            matching_loss_weight
        )

    def _retrieval_loss(
        self,
        image_embeddings,
        text_embeddings,
    ):
        image_embeddings = tf.convert_to_tensor(
            image_embeddings
        )
        text_embeddings = tf.convert_to_tensor(
            text_embeddings
        )

        tf.debugging.assert_rank(image_embeddings, 2)
        tf.debugging.assert_rank(text_embeddings, 3)

        batch_size = tf.shape(image_embeddings)[0]
        image_dim = tf.shape(image_embeddings)[1]

        tf.debugging.assert_greater(
            batch_size,
            1,
            message=(
                "InfoNCE requires at least two distinct images per batch "
                "to provide in-batch negatives. Use batch_size >= 2; "
                "a batch of one produces a degenerate zero retrieval loss."
            ),
        )
        tf.debugging.assert_positive(
            tf.shape(text_embeddings)[1]
        )

        tf.debugging.assert_equal(
            batch_size,
            tf.shape(text_embeddings)[0],
            message="Image and text batch sizes must match.",
        )

        tf.debugging.assert_equal(
            image_dim,
            tf.shape(text_embeddings)[2],
            message="Image and text dimensions must match.",
        )

        image_embeddings = tf.math.l2_normalize(
            tf.cast(image_embeddings, tf.float32),
            axis=-1,
        )

        text_embeddings = tf.math.l2_normalize(
            tf.cast(text_embeddings, tf.float32),
            axis=-1,
        )

        num_captions = tf.shape(text_embeddings)[1]

        flat_text_embeddings = tf.reshape(
            text_embeddings,
            (
                batch_size * num_captions,
                image_dim,
            ),
        )

        logits = tf.matmul(
            image_embeddings,
            flat_text_embeddings,
            transpose_b=True,
        )

        logits = logits / self.temperature

        image_ids = tf.range(batch_size)

        caption_owner_ids = tf.repeat(
            tf.range(batch_size),
            repeats=num_captions,
        )

        positive_mask = tf.equal(
            image_ids[:, tf.newaxis],
            caption_owner_ids[tf.newaxis, :],
        )

        # Image-to-text: all captions belonging to the image
        # are treated as positives.
        image_to_text_log_probs = tf.nn.log_softmax(
            logits,
            axis=1,
        )

        positive_log_probs = tf.where(
            positive_mask,
            image_to_text_log_probs,
            tf.zeros_like(image_to_text_log_probs),
        )

        image_to_text_loss = -tf.reduce_mean(
            tf.reduce_sum(
                positive_log_probs,
                axis=1,
            ) / tf.cast(num_captions, tf.float32)
        )

        # Text-to-image: each caption retrieves its owner image.
        reverse_logits = tf.transpose(logits)

        text_to_image_targets = tf.repeat(
            tf.range(batch_size),
            repeats=num_captions,
        )

        text_to_image_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=text_to_image_targets,
                logits=reverse_logits,
            )
        )

        return (
            image_to_text_loss + text_to_image_loss
        ) / 2.0

    @staticmethod
    def _matching_loss(
        matching_scores,
        negative_matching_scores=None,
        matching_negatives_valid=None,
    ):
        matching_scores = tf.cast(
            matching_scores,
            tf.float32,
        )

        tf.debugging.assert_rank(
            matching_scores,
            2,
            message="matching_scores must have shape [B, C].",
        )

        tf.debugging.assert_all_finite(
            matching_scores,
            "Positive matching scores contain NaN or Inf.",
        )

        # Positive pairs should have high logits.
        positive_loss = tf.reduce_mean(
            tf.nn.softplus(-matching_scores)
        )

        if negative_matching_scores is None:
            return positive_loss

        negative_matching_scores = tf.cast(
            negative_matching_scores,
            tf.float32,
        )

        tf.debugging.assert_equal(
            tf.shape(matching_scores),
            tf.shape(negative_matching_scores),
            message="Positive and negative matching shapes must match.",
        )

        tf.debugging.assert_all_finite(
            negative_matching_scores,
            "Negative matching scores contain NaN or Inf.",
        )

        negative_loss = tf.reduce_mean(
            tf.nn.softplus(negative_matching_scores)
        )

        if matching_negatives_valid is None:
            matching_negatives_valid = True

        matching_negatives_valid = tf.cast(
            matching_negatives_valid,
            tf.bool,
        )

        negative_loss = tf.where(
            matching_negatives_valid,
            negative_loss,
            tf.zeros_like(negative_loss),
        )

        return (
            positive_loss + negative_loss
        ) / 2.0

    def call(
        self,
        image_embeddings,
        text_embeddings,
        matching_scores=None,
        negative_matching_scores=None,
        matching_negatives_valid=None,
        training=None,
    ):
        del training
        retrieval_loss = self._retrieval_loss(
            image_embeddings,
            text_embeddings,
        )

        if matching_scores is None:
            if (
                negative_matching_scores is not None
                or matching_negatives_valid is not None
            ):
                raise ValueError(
                    "Negative matching arguments require matching_scores."
                )

            return retrieval_loss

        matching_loss = self._matching_loss(
            matching_scores=matching_scores,
            negative_matching_scores=negative_matching_scores,
            matching_negatives_valid=matching_negatives_valid,
        )

        total_loss = (
            retrieval_loss
            + self.matching_loss_weight * matching_loss
        )

        tf.debugging.assert_all_finite(
            total_loss,
            "Combined retrieval and matching loss is not finite.",
        )

        return total_loss

    def get_config(self):
        config = super().get_config()

        config.update({
            "temperature": self.temperature,
            "matching_loss_weight": self.matching_loss_weight,
        })

        return config


def _run_infonce_tests():
    tf.random.set_seed(42)

    print("Multi-Positive InfoNCE Loss Tests")
    print("-" * 55)
    print(f"TensorFlow: {tf.__version__}")

    batch_size = 8
    num_captions = 5
    embedding_dim = 512

    image_embeddings = tf.Variable(
        tf.random.normal((batch_size, embedding_dim))
    )

    text_embeddings = tf.Variable(
        tf.random.normal(
            (batch_size, num_captions, embedding_dim)
        )
    )

    matching_scores = tf.Variable(
        tf.random.normal((batch_size, num_captions))
    )

    negative_matching_scores = tf.Variable(
        tf.random.normal((batch_size, num_captions))
    )

    criterion = MultiPositiveInfoNCELoss(
        temperature=0.07,
        matching_loss_weight=0.2,
    )

    with tf.GradientTape() as tape:
        loss = criterion(
            image_embeddings,
            text_embeddings,
            matching_scores=matching_scores,
            negative_matching_scores=negative_matching_scores,
            matching_negatives_valid=True,
        )

    assert loss.shape.rank == 0

    tf.debugging.assert_all_finite(
        loss,
        "Loss is not finite.",
    )

    variables = [
        image_embeddings,
        text_embeddings,
        matching_scores,
        negative_matching_scores,
    ]

    gradients = tape.gradient(loss, variables)

    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Gradient contains NaN or Inf.",
        )

    print("Combined loss scalar: PASS")
    print("Finite loss and gradients: PASS")
    print("Matching branch gradient flow: PASS")

    # A one-image batch has no in-batch negatives and must fail loudly
    # instead of silently returning an apparently perfect zero loss.
    try:
        criterion(
            tf.random.normal((1, embedding_dim)),
            tf.random.normal((1, 2, embedding_dim)),
        )
        raise AssertionError("InfoNCE must reject a batch of one image.")
    except tf.errors.InvalidArgumentError:
        print("Degenerate single-image batch rejection: PASS")

    # Retrieval-only mode remains supported.
    retrieval_only_loss = criterion(
        tf.random.normal((4, embedding_dim)),
        tf.random.normal((4, 2, embedding_dim)),
    )

    tf.debugging.assert_all_finite(
        retrieval_only_loss,
        "Retrieval-only loss is not finite.",
    )

    print("Retrieval-only compatibility: PASS")

    # Invalid temperature.
    try:
        MultiPositiveInfoNCELoss(temperature=0)
        raise AssertionError(
            "Zero temperature should be rejected."
        )
    except ValueError:
        print("Temperature validation: PASS")

    print("-" * 55)
    print("All Multi-Positive InfoNCE tests passed!")


if __name__ == "__main__":
    _run_infonce_tests()
