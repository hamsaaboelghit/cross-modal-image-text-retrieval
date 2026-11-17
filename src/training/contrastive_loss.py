import tensorflow as tf


class MultiPositiveInfoNCELoss(tf.keras.layers.Layer):
    """
    Symmetric multi-positive InfoNCE loss.

    image_embeddings: [B, D]
    text_embeddings:  [B, C, D]

    Each image has C positive captions.
    """

    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)

        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        self.temperature = float(temperature)

    def call(self, image_embeddings, text_embeddings):
        image_embeddings = tf.convert_to_tensor(image_embeddings)
        text_embeddings = tf.convert_to_tensor(text_embeddings)

        tf.debugging.assert_rank(
            image_embeddings, 2,
            message="Image embeddings must have shape [B, D].",
        )

        tf.debugging.assert_rank(
            text_embeddings, 3,
            message="Text embeddings must have shape [B, C, D].",
        )

        batch_size = tf.shape(image_embeddings)[0]
        image_dim = tf.shape(image_embeddings)[1]

        text_batch_size = tf.shape(text_embeddings)[0]
        num_captions = tf.shape(text_embeddings)[1]
        text_dim = tf.shape(text_embeddings)[2]

        tf.debugging.assert_positive(batch_size)
        tf.debugging.assert_positive(num_captions)

        tf.debugging.assert_equal(
            batch_size,
            text_batch_size,
            message="Image and text batch sizes must match.",
        )

        tf.debugging.assert_equal(
            image_dim,
            text_dim,
            message="Image and text embedding dimensions must match.",
        )

        image_embeddings = tf.math.l2_normalize(
            tf.cast(image_embeddings, tf.float32),
            axis=-1,
        )

        text_embeddings = tf.math.l2_normalize(
            tf.cast(text_embeddings, tf.float32),
            axis=-1,
        )

        embedding_dim = tf.shape(text_embeddings)[-1]

        flat_text_embeddings = tf.reshape(
            text_embeddings,
            (batch_size * num_captions, embedding_dim),
        )

        # Image-to-text similarity: [B, B*C].
        logits = tf.matmul(
            image_embeddings,
            flat_text_embeddings,
            transpose_b=True,
        )

        logits = logits / self.temperature

        # Each image has C positive captions.
        image_ids = tf.range(batch_size)
        caption_owner_ids = tf.repeat(
            tf.range(batch_size),
            repeats=num_captions,
        )

        positive_mask = tf.equal(
            image_ids[:, tf.newaxis],
            caption_owner_ids[tf.newaxis, :],
        )

        # Image-to-text multi-positive loss.
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
            tf.reduce_sum(positive_log_probs, axis=1)
            / tf.cast(num_captions, tf.float32)
        )

        # Text-to-image loss.
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

        # Symmetric loss.
        loss = (
            image_to_text_loss + text_to_image_loss
        ) / 2.0

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
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

    criterion = MultiPositiveInfoNCELoss(
        temperature=0.07
    )

    with tf.GradientTape() as tape:
        loss = criterion(
            image_embeddings,
            text_embeddings,
        )

    assert loss.shape.rank == 0
    tf.debugging.assert_all_finite(loss, "Loss is not finite.")

    gradients = tape.gradient(
        loss,
        [image_embeddings, text_embeddings],
    )

    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Gradient contains NaN or Inf.",
        )

    print(f"Loss: {loss.numpy():.4f}")
    print("Scalar loss: PASS")
    print("Finite loss: PASS")
    print("Gradient flow: PASS")

    # Verify that identical positive embeddings produce a lower
    # loss than randomly mismatched embeddings.
    perfect_images = tf.eye(4)
    perfect_texts = tf.repeat(
        perfect_images[:, tf.newaxis, :],
        repeats=2,
        axis=1,
    )

    perfect_loss = criterion(
        perfect_images,
        perfect_texts,
    )

    random_loss = criterion(
        tf.random.normal((4, 4)),
        tf.random.normal((4, 2, 4)),
    )

    assert tf.math.is_finite(perfect_loss)
    assert tf.math.is_finite(random_loss)

    print("Positive-pair behavior: PASS")

    # Invalid temperature.
    try:
        MultiPositiveInfoNCELoss(temperature=0)
        raise AssertionError("Zero temperature should be rejected.")
    except ValueError:
        print("Temperature validation: PASS")

    # Invalid batch mismatch.
    try:
        criterion(
            tf.random.normal((3, 8)),
            tf.random.normal((4, 2, 8)),
        )
        raise AssertionError("Batch mismatch should be rejected.")
    except tf.errors.InvalidArgumentError:
        print("Batch-size validation: PASS")

    print("-" * 55)
    print("All Multi-Positive InfoNCE tests passed!")


if __name__ == "__main__":
    _run_infonce_tests()