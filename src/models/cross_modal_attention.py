
import tensorflow as tf


class CrossModalAttention(tf.keras.layers.Layer):
    """
    Text-to-Image Multi-Head Cross-Attention.

    Text tokens act as Queries.
    Image regions act as Keys and Values.

    Inputs:
        text_features: [B, L, text_dim]
        image_regions: [B, R, image_dim]
        text_mask: [B, L], True for valid tokens

    Outputs:
        output_features: [B, L, embedding_dim]
        attention_weights: [B, L, R]
    """

    def __init__(
        self,
        text_dim,
        image_dim=2048,
        embedding_dim=512,
        num_heads=8,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_projection = tf.keras.layers.Dense(
            embedding_dim,
            name="query_projection",
        )

        self.key_projection = tf.keras.layers.Dense(
            embedding_dim,
            name="key_projection",
        )

        self.value_projection = tf.keras.layers.Dense(
            embedding_dim,
            name="value_projection",
        )

        self.output_projection = tf.keras.layers.Dense(
            embedding_dim,
            name="output_projection",
        )

        self.residual_projection = tf.keras.layers.Dense(
            embedding_dim,
            name="residual_projection",
        )

        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.output_dropout = tf.keras.layers.Dropout(dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="output_layer_norm",
        )

    def call(
        self,
        text_features,
        image_regions,
        text_mask=None,
        training=None,
    ):
        text_features = tf.convert_to_tensor(text_features)
        image_regions = tf.convert_to_tensor(image_regions)

        tf.debugging.assert_rank(
            text_features,
            3,
            message="text_features must have shape [B, L, text_dim].",
        )

        tf.debugging.assert_rank(
            image_regions,
            3,
            message="image_regions must have shape [B, R, image_dim].",
        )

        tf.debugging.assert_equal(
            tf.shape(text_features)[-1],
            self.text_dim,
            message="Unexpected text feature dimension.",
        )

        tf.debugging.assert_equal(
            tf.shape(image_regions)[-1],
            self.image_dim,
            message="Unexpected image feature dimension.",
        )

        tf.debugging.assert_equal(
            tf.shape(text_features)[0],
            tf.shape(image_regions)[0],
            message="Text and image batches must have the same size.",
        )

        tf.debugging.assert_positive(
            tf.shape(image_regions)[1],
            message="At least one image region is required.",
        )

        batch_size = tf.shape(text_features)[0]
        sequence_length = tf.shape(text_features)[1]
        num_regions = tf.shape(image_regions)[1]

        if text_mask is not None:
            text_mask = tf.convert_to_tensor(text_mask)

            if text_mask.dtype != tf.bool:
                raise TypeError("text_mask must have boolean dtype.")

            tf.debugging.assert_equal(
                tf.shape(text_mask),
                tf.shape(text_features)[:2],
                message="text_mask must have shape [B, L].",
            )

            tf.debugging.assert_equal(
                tf.reduce_all(
                    tf.reduce_any(text_mask, axis=1)
                ),
                True,
                message="Every sample must contain at least one valid token.",
            )

        # 1. Project Queries, Keys and Values.
        queries = self.query_projection(text_features)
        keys = self.key_projection(image_regions)
        values = self.value_projection(image_regions)

        # 2. Split into multiple heads.
        # [B, L, E] -> [B, H, L, D]
        # [B, R, E] -> [B, H, R, D]
        queries = tf.reshape(
            queries,
            (batch_size, sequence_length, self.num_heads, self.head_dim),
        )

        keys = tf.reshape(
            keys,
            (batch_size, num_regions, self.num_heads, self.head_dim),
        )

        values = tf.reshape(
            values,
            (batch_size, num_regions, self.num_heads, self.head_dim),
        )

        queries = tf.transpose(queries, (0, 2, 1, 3))
        keys = tf.transpose(keys, (0, 2, 1, 3))
        values = tf.transpose(values, (0, 2, 1, 3))

        # 3. Scaled dot-product attention.
        attention_scores = tf.matmul(
            queries,
            keys,
            transpose_b=True,
        )

        scale = tf.math.sqrt(
            tf.cast(self.head_dim, attention_scores.dtype)
        )

        attention_scores = attention_scores / scale

        attention_weights = tf.nn.softmax(
            attention_scores,
            axis=-1,
        )

        normalized_attention_weights = attention_weights

        dropped_weights = self.attention_dropout(
            attention_weights,
            training=training,
        )

        # 4. Aggregate image information for each text token.
        context = tf.matmul(
            dropped_weights,
            values,
        )

        # [B, H, L, D] -> [B, L, E]
        context = tf.transpose(context, (0, 2, 1, 3))

        context = tf.reshape(
            context,
            (batch_size, sequence_length, self.embedding_dim),
        )

        # 5. Output projection and residual connection.
        context = self.output_projection(context)

        residual = self.residual_projection(text_features)

        output_features = self.layer_norm(
            residual
            + self.output_dropout(
                context,
                training=training,
            )
        )

        # 6. Zero padded text positions.
        if text_mask is not None:
            valid_positions = text_mask[:, :, tf.newaxis]

            output_features = tf.where(
                valid_positions,
                output_features,
                tf.zeros_like(output_features),
            )

            normalized_attention_weights = tf.where(
                text_mask[:, tf.newaxis, :, tf.newaxis],
                normalized_attention_weights,
                tf.zeros_like(normalized_attention_weights),
            )

        # Average normalized weights across attention heads.
        # [B, H, L, R] -> [B, L, R]
        averaged_attention = tf.reduce_mean(
            normalized_attention_weights,
            axis=1,
        )

        return output_features, averaged_attention


def _run_cross_modal_attention_tests():
    tf.random.set_seed(42)

    print("Cross-Modal Attention Tests")
    print("-" * 60)
    print(f"TensorFlow: {tf.__version__}")

    batch_size = 4
    sequence_length = 34
    num_regions = 49
    image_dim = 2048
    embedding_dim = 512

    for text_dim in [512, 1024]:
        print(f"\nTesting text_dim={text_dim}")

        model = CrossModalAttention(
            text_dim=text_dim,
            image_dim=image_dim,
            embedding_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
        )

        text_features = tf.random.normal(
            (batch_size, sequence_length, text_dim)
        )

        image_regions = tf.random.normal(
            (batch_size, num_regions, image_dim)
        )

        text_mask = tf.sequence_mask(
            [34, 28, 20, 15],
            maxlen=sequence_length,
        )

        # Evaluation mode.
        output_features, attention_weights = model(
            text_features,
            image_regions,
            text_mask=text_mask,
            training=False,
        )

        assert tuple(output_features.shape) == (
            batch_size,
            sequence_length,
            embedding_dim,
        )

        assert tuple(attention_weights.shape) == (
            batch_size,
            sequence_length,
            num_regions,
        )

        assert bool(tf.reduce_all(tf.math.is_finite(output_features)))
        assert bool(tf.reduce_all(tf.math.is_finite(attention_weights)))

        attention_sums = tf.reduce_sum(
            attention_weights,
            axis=-1,
        )

        valid_sums = tf.boolean_mask(
            attention_sums,
            text_mask,
        )

        tf.debugging.assert_near(
            valid_sums,
            tf.ones_like(valid_sums),
            atol=1e-5,
        )

        padded_attention = tf.boolean_mask(
            attention_weights,
            tf.logical_not(text_mask),
        )

        tf.debugging.assert_near(
            padded_attention,
            tf.zeros_like(padded_attention),
            atol=1e-6,
        )

        padded_outputs = tf.boolean_mask(
            output_features,
            tf.logical_not(text_mask),
        )

        tf.debugging.assert_near(
            padded_outputs,
            tf.zeros_like(padded_outputs),
            atol=1e-6,
        )

        print("Evaluation Mode: PASS")
        print("Attention Normalization: PASS")
        print("Padding Mask: PASS")

        # Training mode and gradient flow.
        with tf.GradientTape() as tape:
            train_outputs, train_weights = model(
                text_features,
                image_regions,
                text_mask=text_mask,
                training=True,
            )

            loss = tf.reduce_mean(tf.square(train_outputs))

        gradients = tape.gradient(
            loss,
            model.trainable_variables,
        )

        assert gradients
        assert all(gradient is not None for gradient in gradients)

        assert bool(
            tf.reduce_all(
                tf.stack(
                    [
                        tf.reduce_all(tf.math.is_finite(gradient))
                        for gradient in gradients
                    ]
                )
            )
        )

        train_sums = tf.reduce_sum(train_weights, axis=-1)
        valid_train_sums = tf.boolean_mask(train_sums, text_mask)

        tf.debugging.assert_near(
            valid_train_sums,
            tf.ones_like(valid_train_sums),
            atol=1e-5,
        )

        print("Training Mode: PASS")
        print("Backward Pass: PASS")
        print(
            f"Output: {tuple(train_outputs.shape)} | "
            f"Attention: {tuple(train_weights.shape)} | PASS"
        )

    # Reject samples with no valid text tokens.
    invalid_mask = tf.zeros(
        (2, 5),
        dtype=tf.bool,
    )

    try:
        model(
            tf.random.normal((2, 5, 1024)),
            tf.random.normal((2, 7, image_dim)),
            text_mask=invalid_mask,
            training=False,
        )
        raise AssertionError("All-padding text should have been rejected")
    except tf.errors.InvalidArgumentError:
        print("All-padding input rejection: PASS")

    print("-" * 60)
    print("All Cross-Modal Attention tests passed!")


if __name__ == "__main__":
    _run_cross_modal_attention_tests()
