import tensorflow as tf


PAD_IDX = 0


class BaselineImageEncoder(tf.keras.Model):
    """
    Lightweight CNN baseline image encoder.

    Input:
        RGB images [B, H, W, 3] in [0, 255].

    Output:
        L2-normalized image embeddings [B, output_dim].
    """

    def __init__(self, output_dim=512, **kwargs):
        super().__init__(**kwargs)

        self.features = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, kernel_size=3, strides=2,
                    padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2D(
                    64, kernel_size=3, strides=2,
                    padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2D(
                    128, kernel_size=3, strides=2,
                    padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2D(
                    256, kernel_size=3, strides=2,
                    padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.GlobalAveragePooling2D(),
            ],
            name="baseline_cnn",
        )

        self.output_projection = tf.keras.layers.Dense(
            output_dim,
            name="image_projection",
        )

        self.output_dim = output_dim

    def call(self, images, training=None):
        images = tf.convert_to_tensor(images)

        tf.debugging.assert_rank(
            images,
            4,
            message="Images must have shape [B, H, W, 3].",
        )

        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must have 3 RGB channels.",
        )

        tf.debugging.assert_positive(
            tf.shape(images)[0],
            message="Image batch cannot be empty.",
        )

        images = tf.cast(images, tf.float32)
        images = images / 255.0

        features = self.features(
            images,
            training=training,
        )

        embeddings = self.output_projection(features)

        embeddings = tf.math.l2_normalize(
            embeddings,
            axis=-1,
        )

        return embeddings


class BaselineTextEncoder(tf.keras.Model):
    """
    Mean-pooled embedding baseline text encoder.

    Input:
        Integer token IDs [N, L].
        PAD token ID = 0.

    Output:
        L2-normalized text embeddings [N, output_dim].
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        output_dim=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vocab_size <= PAD_IDX:
            raise ValueError("vocab_size must be greater than PAD_IDX.")

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=False,
            name="token_embedding",
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.output_projection = tf.keras.layers.Dense(
            output_dim,
            name="text_projection",
        )

        self.output_dim = output_dim

    def call(self, captions, training=None):
        captions = tf.convert_to_tensor(captions)

        tf.debugging.assert_rank(
            captions,
            2,
            message="Captions must have shape [N, L].",
        )

        tf.debugging.assert_positive(
            tf.shape(captions)[0],
            message="Caption batch cannot be empty.",
        )

        tf.debugging.assert_positive(
            tf.shape(captions)[1],
            message="Caption sequence length cannot be zero.",
        )

        mask = tf.not_equal(captions, PAD_IDX)

        lengths = tf.reduce_sum(
            tf.cast(mask, tf.float32),
            axis=1,
            keepdims=True,
        )

        tf.debugging.assert_positive(
            lengths,
            message="Every caption must contain at least one token.",
        )

        embeddings = self.embedding(captions)

        mask = tf.cast(
            mask[:, :, tf.newaxis],
            embeddings.dtype,
        )

        embeddings = embeddings * mask

        pooled = tf.reduce_sum(
            embeddings,
            axis=1,
        ) / tf.cast(lengths, embeddings.dtype)

        pooled = self.dropout(
            pooled,
            training=training,
        )

        projected = self.output_projection(pooled)

        projected = tf.math.l2_normalize(
            projected,
            axis=-1,
        )

        return projected


class BaselineRetrievalModel(tf.keras.Model):
    """
    Baseline image-text retrieval model.

    Images:
        [B, H, W, 3]

    Captions:
        [B, C, L]

    Outputs:
        image_embeddings: [B, output_dim]
        text_embeddings: [B, C, output_dim]
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        output_dim=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_encoder = BaselineImageEncoder(
            output_dim=output_dim,
            name="baseline_image_encoder",
        )

        self.text_encoder = BaselineTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            dropout=dropout,
            name="baseline_text_encoder",
        )

        self.embedding_dim = output_dim

    def call(self, images, captions, training=None):
        images = tf.convert_to_tensor(images)
        captions = tf.convert_to_tensor(captions)

        tf.debugging.assert_rank(
            images,
            4,
            message="Images must have shape [B, H, W, 3].",
        )

        tf.debugging.assert_rank(
            captions,
            3,
            message="Captions must have shape [B, C, L].",
        )

        batch_size = tf.shape(images)[0]
        caption_batch_size = tf.shape(captions)[0]
        num_captions = tf.shape(captions)[1]
        sequence_length = tf.shape(captions)[2]

        tf.debugging.assert_equal(
            batch_size,
            caption_batch_size,
            message="Images and captions must have the same batch size.",
        )

        tf.debugging.assert_positive(
            batch_size,
            message="Batch cannot be empty.",
        )

        tf.debugging.assert_positive(
            num_captions,
            message="At least one caption per image is required.",
        )

        tf.debugging.assert_positive(
            sequence_length,
            message="Caption sequence length cannot be zero.",
        )

        image_embeddings = self.image_encoder(
            images,
            training=training,
        )

        flat_captions = tf.reshape(
            captions,
            (batch_size * num_captions, sequence_length),
        )

        flat_text_embeddings = self.text_encoder(
            flat_captions,
            training=training,
        )

        text_embeddings = tf.reshape(
            flat_text_embeddings,
            (batch_size, num_captions, self.embedding_dim),
        )

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


def _run_baseline_tests():
    tf.random.set_seed(42)

    print("Baseline Model Tests")
    print("-" * 60)
    print(f"TensorFlow: {tf.__version__}")

    batch_size = 2
    num_captions = 3
    sequence_length = 12
    vocab_size = 1000
    output_dim = 512

    images = tf.random.uniform(
        (batch_size, 224, 224, 3),
        minval=0,
        maxval=256,
        dtype=tf.float32,
    )

    captions = tf.random.uniform(
        (batch_size, num_captions, sequence_length),
        minval=4,
        maxval=vocab_size,
        dtype=tf.int32,
    )

    captions_array = captions.numpy()
    captions_array[0, 1, 8:] = PAD_IDX
    captions_array[1, 2, 5:] = PAD_IDX
    captions = tf.convert_to_tensor(captions_array)

    model = BaselineRetrievalModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        output_dim=output_dim,
        dropout=0.1,
    )

    outputs = model(
        images,
        captions,
        training=False,
    )

    assert tuple(outputs["image_embeddings"].shape) == (
        batch_size,
        output_dim,
    )

    assert tuple(outputs["text_embeddings"].shape) == (
        batch_size,
        num_captions,
        output_dim,
    )

    print("Image embedding shape: PASS")
    print("Text embedding shape: PASS")

    image_norms = tf.norm(
        outputs["image_embeddings"],
        axis=-1,
    )

    text_norms = tf.norm(
        outputs["text_embeddings"],
        axis=-1,
    )

    tf.debugging.assert_near(
        image_norms,
        tf.ones((batch_size,)),
        atol=1e-5,
    )

    tf.debugging.assert_near(
        text_norms,
        tf.ones((batch_size, num_captions)),
        atol=1e-5,
    )

    print("Embedding normalization: PASS")

    # Check that padding tokens do not affect mean pooling.
    text_encoder = model.text_encoder

    caption_a = tf.constant([[4, 5, 6, 0, 0]], dtype=tf.int32)
    caption_b = tf.constant([[4, 5, 6, 0, 0, 0, 0]], dtype=tf.int32)

    embedding_a = text_encoder(caption_a, training=False)
    embedding_b = text_encoder(caption_b, training=False)

    tf.debugging.assert_near(
        embedding_a,
        embedding_b,
        atol=1e-5,
    )

    print("Padding invariance: PASS")

    # Check that all-padding captions are rejected.
    all_padding = tf.zeros((2, 5), dtype=tf.int32)

    try:
        text_encoder(all_padding, training=False)
        raise AssertionError("All-padding captions should be rejected.")
    except tf.errors.InvalidArgumentError:
        print("All-padding rejection: PASS")

    # Check gradient flow.
    with tf.GradientTape() as tape:
        train_outputs = model(
            images,
            captions,
            training=True,
        )

        loss = (
            tf.reduce_mean(train_outputs["image_embeddings"])
            + tf.reduce_mean(train_outputs["text_embeddings"])
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables,
    )

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    print("Gradient flow: PASS")

    print("-" * 60)
    print("All Baseline Model tests passed!")


if __name__ == "__main__":
    _run_baseline_tests()