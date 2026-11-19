import tensorflow as tf


class TripletContrastiveLoss(tf.keras.layers.Layer):
    """
    Triplet margin loss for image-text retrieval.

    image_embeddings: [B, D]
    text_embeddings:  [B, C, D]

    One positive caption is selected for each image.
    A negative caption is sampled from a different image.
    """

    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)

        if margin <= 0:
            raise ValueError("Margin must be positive.")

        self.margin = float(margin)

    def call(
        self,
        image_embeddings,
        text_embeddings,
        positive_caption_indices=None,
        training=None,
    ):
        image_embeddings = tf.convert_to_tensor(image_embeddings)
        text_embeddings = tf.convert_to_tensor(text_embeddings)

        tf.debugging.assert_rank(
            image_embeddings,
            2,
            message="Image embeddings must have shape [B, D].",
        )

        tf.debugging.assert_rank(
            text_embeddings,
            3,
            message="Text embeddings must have shape [B, C, D].",
        )

        batch_size = tf.shape(image_embeddings)[0]
        image_dim = tf.shape(image_embeddings)[1]

        text_batch_size = tf.shape(text_embeddings)[0]
        num_captions = tf.shape(text_embeddings)[1]
        text_dim = tf.shape(text_embeddings)[2]

        tf.debugging.assert_greater_equal(
            batch_size,
            2,
            message="Triplet loss requires at least 2 images.",
        )

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

        tf.debugging.assert_positive(
            num_captions,
            message="Each image must have at least one caption.",
        )

        image_embeddings = tf.math.l2_normalize(
            tf.cast(image_embeddings, tf.float32),
            axis=-1,
        )

        text_embeddings = tf.math.l2_normalize(
            tf.cast(text_embeddings, tf.float32),
            axis=-1,
        )

        # Select one positive caption for each image.
        if positive_caption_indices is None:
            if training is False:
                positive_caption_indices = tf.zeros((batch_size,), dtype=tf.int32)
            else:
                positive_caption_indices = tf.random.uniform(
                    shape=(batch_size,), minval=0, maxval=num_captions,
                    dtype=tf.int32,
                )
        else:
            positive_caption_indices = tf.convert_to_tensor(
                positive_caption_indices,
                dtype=tf.int32,
            )

            tf.debugging.assert_rank(
                positive_caption_indices,
                1,
                message="Positive indices must have shape [B].",
            )

            tf.debugging.assert_equal(
                tf.shape(positive_caption_indices)[0],
                batch_size,
                message="One positive index is required per image.",
            )

            tf.debugging.assert_greater_equal(
                positive_caption_indices,
                0,
                message="Positive caption index cannot be negative.",
            )

            tf.debugging.assert_less(
                positive_caption_indices,
                num_captions,
                message="Positive caption index out of range.",
            )

        batch_indices = tf.range(batch_size)

        positive_indices = tf.stack(
            [
                batch_indices,
                positive_caption_indices,
            ],
            axis=1,
        )

        positive_embeddings = tf.gather_nd(
            text_embeddings,
            positive_indices,
        )

        # Select a negative image guaranteed to be different.
        if training is False:
            negative_image_indices = (batch_indices + 1) % batch_size
            negative_caption_indices = tf.zeros((batch_size,), dtype=tf.int32)
        else:
            offsets = tf.random.uniform(
                shape=(batch_size,), minval=1, maxval=batch_size,
                dtype=tf.int32,
            )
            negative_image_indices = (batch_indices + offsets) % batch_size
            negative_caption_indices = tf.random.uniform(
                shape=(batch_size,), minval=0, maxval=num_captions,
                dtype=tf.int32,
            )

        negative_indices = tf.stack(
            [
                negative_image_indices,
                negative_caption_indices,
            ],
            axis=1,
        )

        negative_embeddings = tf.gather_nd(
            text_embeddings,
            negative_indices,
        )

        # Euclidean distances.
        positive_distances = tf.norm(
            image_embeddings - positive_embeddings,
            axis=-1,
        )

        negative_distances = tf.norm(
            image_embeddings - negative_embeddings,
            axis=-1,
        )

        # Triplet margin objective.
        losses = tf.nn.relu(
            positive_distances
            - negative_distances
            + self.margin
        )

        return tf.reduce_mean(losses)

    def get_config(self):
        config = super().get_config()
        config.update({
            "margin": self.margin,
        })
        return config


def _run_triplet_tests():
    tf.random.set_seed(42)

    print("Triplet Loss Tests")
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

    criterion = TripletContrastiveLoss(margin=0.2)

    with tf.GradientTape() as tape:
        loss = criterion(
            image_embeddings,
            text_embeddings,
        )

    assert loss.shape.rank == 0
    tf.debugging.assert_all_finite(loss, "Loss is not finite.")
    tf.debugging.assert_greater_equal(loss, 0.0)

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
    print("Non-negative loss: PASS")
    print("Gradient flow: PASS")

    # Explicit positive indices.
    positive_indices = tf.constant(
        [0, 1, 2, 3, 4, 0, 1, 2],
        dtype=tf.int32,
    )

    indexed_loss = criterion(
        image_embeddings,
        text_embeddings,
        positive_caption_indices=positive_indices,
    )

    tf.debugging.assert_all_finite(
        indexed_loss,
        "Indexed loss is not finite.",
    )

    print("Explicit positive indices: PASS")

    # Invalid positive indices.
    try:
        criterion(
            image_embeddings,
            text_embeddings,
            positive_caption_indices=tf.fill(
                (batch_size,),
                num_captions,
            ),
        )
        raise AssertionError("Out-of-range indices should be rejected.")
    except tf.errors.InvalidArgumentError:
        print("Positive index validation: PASS")

    # Batch size one must fail.
    try:
        criterion(
            tf.random.normal((1, embedding_dim)),
            tf.random.normal((1, num_captions, embedding_dim)),
        )
        raise AssertionError("Batch size one should be rejected.")
    except tf.errors.InvalidArgumentError:
        print("Minimum batch-size validation: PASS")

    # Verify negative indices never select the anchor image.
    for test_batch_size in (2, 4, 8):
        indices = tf.range(test_batch_size)

        offsets = tf.random.uniform(
            (test_batch_size,),
            minval=1,
            maxval=test_batch_size,
            dtype=tf.int32,
        )

        negative_indices = (
            indices + offsets
        ) % test_batch_size

        assert bool(
            tf.reduce_all(indices != negative_indices).numpy()
        )

    print("Negative sampling checks: PASS")

    # Invalid margin.
    try:
        TripletContrastiveLoss(margin=0)
        raise AssertionError("Zero margin should be rejected.")
    except ValueError:
        print("Margin validation: PASS")

    print("-" * 55)
    print("All Triplet Loss tests passed!")


if __name__ == "__main__":
    _run_triplet_tests()