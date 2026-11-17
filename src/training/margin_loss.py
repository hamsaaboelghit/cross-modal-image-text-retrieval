import tensorflow as tf


class MarginBasedContrastiveLoss(tf.keras.layers.Layer):
    """
    Pairwise Contrastive Loss for image-text retrieval.

    Image embeddings: [B, D]
    Text embeddings:  [B, C, D]

    Positive pairs:
        Each image with its own valid captions.

    Negative pairs:
        Each image with captions belonging to other images.
    """

    def __init__(self, margin=1.0, name="margin_contrastive_loss"):
        super().__init__(name=name)

        if not isinstance(margin, (int, float)):
            raise TypeError("Margin must be a number.")

        if not 0 < margin <= 2:
            raise ValueError(
                "Margin must be greater than 0 and at most 2 "
                "for normalized embeddings."
            )

        self.margin = float(margin)

    def call(
        self,
        image_embeddings,
        text_embeddings,
        caption_mask=None
    ):
        image_embeddings = tf.convert_to_tensor(image_embeddings)
        text_embeddings = tf.convert_to_tensor(text_embeddings)

        if image_embeddings.shape.rank != 2:
            raise ValueError(
                "image_embeddings must have shape [B, D]."
            )

        if text_embeddings.shape.rank != 3:
            raise ValueError(
                "text_embeddings must have shape [B, C, D]."
            )

        tf.debugging.assert_equal(
            tf.shape(image_embeddings)[0],
            tf.shape(text_embeddings)[0],
            message="Image and text batch sizes must match."
        )

        tf.debugging.assert_equal(
            tf.shape(image_embeddings)[1],
            tf.shape(text_embeddings)[2],
            message="Image and text embedding dimensions must match."
        )

        tf.debugging.assert_greater_equal(
            tf.shape(image_embeddings)[0],
            2,
            message="Batch size must be at least 2."
        )

        tf.debugging.assert_greater_equal(
            tf.shape(text_embeddings)[1],
            1,
            message="At least one caption per image is required."
        )

        if not image_embeddings.dtype.is_floating:
            raise TypeError("Image embeddings must be floating-point.")

        if not text_embeddings.dtype.is_floating:
            raise TypeError("Text embeddings must be floating-point.")

        image_embeddings = tf.math.l2_normalize(
            tf.cast(image_embeddings, tf.float32),
            axis=-1
        )

        text_embeddings = tf.math.l2_normalize(
            tf.cast(text_embeddings, tf.float32),
            axis=-1
        )

        batch_size = tf.shape(image_embeddings)[0]
        num_captions = tf.shape(text_embeddings)[1]

        if caption_mask is None:
            caption_mask = tf.ones(
                [batch_size, num_captions],
                dtype=tf.bool
            )
        else:
            caption_mask = tf.convert_to_tensor(
                caption_mask,
                dtype=tf.bool
            )

            tf.debugging.assert_equal(
                tf.shape(caption_mask),
                [batch_size, num_captions],
                message="caption_mask must have shape [B, C]."
            )

        tf.debugging.assert_equal(
            tf.reduce_all(tf.reduce_any(caption_mask, axis=1)),
            True,
            message="Every image must have at least one valid caption."
        )

        text_flat = tf.reshape(
            text_embeddings,
            [-1, tf.shape(image_embeddings)[-1]]
        )

        valid_captions = tf.reshape(caption_mask, [-1])

        distances = tf.norm(
            image_embeddings[:, tf.newaxis, :] -
            text_flat[tf.newaxis, :, :],
            axis=-1
        )

        image_ids = tf.range(batch_size)

        caption_image_ids = tf.repeat(
            tf.range(batch_size),
            repeats=num_captions
        )

        positive_mask = tf.equal(
            image_ids[:, tf.newaxis],
            caption_image_ids[tf.newaxis, :]
        )

        valid_mask = tf.broadcast_to(
            valid_captions[tf.newaxis, :],
            tf.shape(positive_mask)
        )

        positive_mask = tf.logical_and(
            positive_mask,
            valid_mask
        )

        negative_mask = tf.logical_and(
            tf.logical_not(positive_mask),
            valid_mask
        )

        positive_distances = tf.boolean_mask(
            distances,
            positive_mask
        )

        negative_distances = tf.boolean_mask(
            distances,
            negative_mask
        )

        positive_loss = tf.reduce_mean(
            tf.square(positive_distances)
        )

        negative_loss = tf.reduce_mean(
            tf.square(
                tf.nn.relu(self.margin - negative_distances)
            )
        )

        return 0.5 * (positive_loss + negative_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


def test_margin_loss():
    print("Margin-Based Contrastive Loss Tests")
    print("-" * 50)

    criterion = MarginBasedContrastiveLoss(margin=1.0)

    images = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=tf.float32
    )

    captions = tf.constant(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]]
        ],
        dtype=tf.float32
    )

    expected_loss = 0.75
    actual_loss = criterion(images, captions)

    tf.debugging.assert_near(
        actual_loss,
        expected_loss,
        atol=1e-5
    )
    print("Deterministic mathematical test: PASS")

    perfect_images = tf.constant(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=tf.float32
    )

    perfect_captions = tf.repeat(
        perfect_images[:, tf.newaxis, :],
        repeats=2,
        axis=1
    )

    perfect_loss = criterion(
        perfect_images,
        perfect_captions
    )

    tf.debugging.assert_near(
        perfect_loss,
        0.0,
        atol=1e-5
    )
    print("Perfect matching test: PASS")

    images = tf.random.normal([4, 16])
    captions = tf.random.normal([4, 3, 16])

    caption_mask = tf.constant(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [True, True, False]
        ]
    )

    masked_loss = criterion(
        images,
        captions,
        caption_mask=caption_mask
    )

    assert bool(tf.math.is_finite(masked_loss))
    assert float(masked_loss) >= 0
    print("Caption mask test: PASS")

    image_embeddings = tf.Variable(
        tf.random.normal([8, 512])
    )
    text_embeddings = tf.Variable(
        tf.random.normal([8, 5, 512])
    )

    with tf.GradientTape() as tape:
        loss = criterion(
            image_embeddings,
            text_embeddings
        )

    image_grad, text_grad = tape.gradient(
        loss,
        [image_embeddings, text_embeddings]
    )

    assert image_grad is not None
    assert text_grad is not None
    assert bool(tf.reduce_all(tf.math.is_finite(image_grad)))
    assert bool(tf.reduce_all(tf.math.is_finite(text_grad)))
    print("Gradient flow: PASS")

    try:
        criterion(
            image_embeddings[:1],
            text_embeddings[:1]
        )
        raise AssertionError("Expected batch-size validation error.")
    except (ValueError, tf.errors.InvalidArgumentError):
        print("Batch-size validation: PASS")

    try:
        MarginBasedContrastiveLoss(margin=2.5)
        raise AssertionError("Expected invalid-margin error.")
    except ValueError:
        print("Margin validation: PASS")

    print("-" * 50)
    print("All Margin-Based Contrastive Loss tests passed!")


if __name__ == "__main__":
    test_margin_loss()