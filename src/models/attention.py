
import tensorflow as tf


class AdditiveAttention(tf.keras.layers.Layer):
    """
    Additive Attention with padding-mask support.

    Input:
        hidden_states: [B, L, hidden_dim]
        mask:          [B, L], True for valid tokens.

    Returns:
        context_vector:   [B, hidden_dim]
        attention_weights:[B, L]
    """

    def __init__(
        self,
        hidden_dim,
        attention_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if attention_dim is None:
            attention_dim = hidden_dim

        if attention_dim <= 0:
            raise ValueError(
                "attention_dim must be greater than zero."
            )

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.feature_layer = tf.keras.layers.Dense(
            attention_dim,
            activation=None,
            use_bias=True,
        )

        self.score_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=False,
        )

    def call(self, hidden_states, mask=None):
        hidden_states = tf.convert_to_tensor(hidden_states)

        tf.debugging.assert_rank(
            hidden_states,
            3,
            message=(
                "hidden_states must have shape "
                "[batch, sequence_length, hidden_dim]."
            ),
        )

        tf.debugging.assert_equal(
            tf.shape(hidden_states)[-1],
            self.hidden_dim,
            message="Unexpected hidden dimension.",
        )

        tf.debugging.assert_positive(
            tf.shape(hidden_states)[0],
            message="Batch size must be greater than zero.",
        )

        tf.debugging.assert_positive(
            tf.shape(hidden_states)[1],
            message="Sequence length must be greater than zero.",
        )

        tf.debugging.assert_all_finite(
            hidden_states,
            "hidden_states contains NaN or Inf.",
        )

        # Additive attention scores.
        scores = self.score_layer(
            tf.math.tanh(
                self.feature_layer(hidden_states)
            )
        )

        scores = tf.squeeze(scores, axis=-1)
        # scores: [B, L]

        if mask is not None:
            mask = tf.cast(mask, tf.bool)

            tf.debugging.assert_equal(
                tf.shape(mask),
                tf.shape(scores),
                message="mask must have shape [B, L].",
            )

            # Reject samples containing only padding tokens.
            valid_per_sample = tf.reduce_any(
                mask,
                axis=1,
            )

            tf.debugging.assert_equal(
                tf.reduce_all(valid_per_sample),
                True,
                message=(
                    "Every sample must contain at least "
                    "one valid token. All-padding sequences "
                    "are not supported."
                ),
            )

            # Exclude padding positions from softmax.
            scores = tf.where(
                mask,
                scores,
                tf.cast(-1e9, scores.dtype),
            )

        # Normalize attention scores.
        attention_weights = tf.nn.softmax(
            scores,
            axis=1,
        )

        # Context vector: weighted sum of token features.
        context_vector = tf.reduce_sum(
            hidden_states
            * tf.expand_dims(attention_weights, axis=-1),
            axis=1,
        )

        return context_vector, attention_weights


if __name__ == "__main__":
    print("Additive Attention Tests")
    print("-" * 45)

    batch_size = 4
    sequence_length = 34
    hidden_dim = 512

    model = AdditiveAttention(
        hidden_dim=hidden_dim,
    )

    hidden_states = tf.random.normal(
        shape=(
            batch_size,
            sequence_length,
            hidden_dim,
        )
    )

    lengths = tf.constant(
        [34, 28, 25, 20],
        dtype=tf.int32,
    )

    positions = tf.range(
        sequence_length,
        dtype=tf.int32,
    )[tf.newaxis, :]

    mask = positions < lengths[:, tf.newaxis]

    # Test 1: Forward pass
    with tf.GradientTape() as tape:
        tape.watch(hidden_states)

        context, weights = model(
            hidden_states,
            mask=mask,
        )

        loss = tf.reduce_mean(tf.square(context))

    assert tuple(context.shape) == (
        batch_size,
        hidden_dim,
    )

    assert tuple(weights.shape) == (
        batch_size,
        sequence_length,
    )

    tf.debugging.assert_all_finite(
        context,
        "Context contains NaN or Inf.",
    )

    tf.debugging.assert_all_finite(
        weights,
        "Attention weights contain NaN or Inf.",
    )

    print("Forward pass: PASS")

    # Test 2: Attention normalization
    weight_sums = tf.reduce_sum(
        weights,
        axis=1,
    )

    tf.debugging.assert_near(
        weight_sums,
        tf.ones((batch_size,)),
        atol=1e-6,
    )

    print("Attention normalization: PASS")

    # Test 3: Padding mask
    invalid_positions = tf.logical_not(mask)

    padded_weights = tf.boolean_mask(
        weights,
        invalid_positions,
    )

    tf.debugging.assert_near(
        padded_weights,
        tf.zeros_like(padded_weights),
        atol=1e-7,
    )

    print("Padding mask: PASS")

    # Test 4: Backward pass
    gradients = tape.gradient(
        loss,
        [
            hidden_states,
            *model.trainable_variables,
        ],
    )

    assert all(
        gradient is not None
        for gradient in gradients
    )

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "A gradient contains NaN or Inf.",
        )

    print("Backward pass: PASS")

    # Test 5: All-padding input rejection
    all_padding_mask = tf.zeros(
        (batch_size, sequence_length),
        dtype=tf.bool,
    )

    try:
        model(
            hidden_states,
            mask=all_padding_mask,
        )

        raise AssertionError(
            "All-padding input should have been rejected."
        )

    except tf.errors.InvalidArgumentError:
        print("All-padding input rejection: PASS")

    # Test 6: No mask
    context_without_mask, weights_without_mask = model(
        hidden_states,
        mask=None,
    )

    assert tuple(context_without_mask.shape) == (
        batch_size,
        hidden_dim,
    )

    assert tuple(weights_without_mask.shape) == (
        batch_size,
        sequence_length,
    )

    print("Forward pass without mask: PASS")

    print("-" * 45)
    print("All Additive Attention tests passed!")
