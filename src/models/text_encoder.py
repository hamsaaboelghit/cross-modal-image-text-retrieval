
import math

import tensorflow as tf

try:
    from .attention import AdditiveAttention
except ImportError:
    from attention import AdditiveAttention


PAD_IDX = 0


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, hidden_dim, dropout=0.1, max_length=512, **kwargs):
        super().__init__(**kwargs)

        if hidden_dim % 2 != 0:
            raise ValueError(
                "hidden_dim must be even for sinusoidal positional encoding"
            )

        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.dropout = tf.keras.layers.Dropout(dropout)

        positions = tf.cast(
            tf.range(max_length)[:, tf.newaxis],
            tf.float32,
        )

        dimensions = tf.cast(
            tf.range(0, hidden_dim, 2)[tf.newaxis, :],
            tf.float32,
        )

        div_term = tf.exp(
            dimensions * (-math.log(10000.0) / hidden_dim)
        )

        angles = positions * div_term

        positional_encoding = tf.stack(
            [tf.sin(angles), tf.cos(angles)],
            axis=-1,
        )

        positional_encoding = tf.reshape(
            positional_encoding,
            (max_length, hidden_dim),
        )

        self.positional_encoding = positional_encoding[tf.newaxis, ...]

    def call(self, embeddings, training=None):
        sequence_length = tf.shape(embeddings)[1]

        tf.debugging.assert_less_equal(
            sequence_length,
            self.max_length,
            message="Sequence length exceeds maximum positional encoding length",
        )

        positions = self.positional_encoding[:, :sequence_length, :]
        embeddings = embeddings + tf.cast(positions, embeddings.dtype)

        return self.dropout(embeddings, training=training)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block with padding-aware self-attention."""

    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )

        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
        )

        self.attention_dropout = tf.keras.layers.Dropout(dropout)
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim * 4, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(hidden_dim),
            ]
        )

        self.output_dropout = tf.keras.layers.Dropout(dropout)
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, valid_mask, training=None):
        # Shape: [batch, 1, sequence_length].
        # True indicates a valid key position.
        attention_mask = valid_mask[:, tf.newaxis, :]

        attention_output = self.self_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
            training=training,
        )

        attention_output = self.attention_dropout(
            attention_output,
            training=training,
        )

        x = self.attention_norm(inputs + attention_output)

        feed_forward_output = self.feed_forward(
            x,
            training=training,
        )

        feed_forward_output = self.output_dropout(
            feed_forward_output,
            training=training,
        )

        outputs = self.output_norm(x + feed_forward_output)

        # Padded query positions must not contribute to later pooling.
        outputs = tf.where(
            valid_mask[:, :, tf.newaxis],
            outputs,
            tf.zeros_like(outputs),
        )

        return outputs


class TextEncoder(tf.keras.Model):
    """
    Text encoder supporting RNN, LSTM, GRU, and Transformer.

    Input:
        captions: integer tensor [batch, sequence_length]
        PAD_IDX is 0.

    Default output:
        features: [batch, output_dim]

    Optional outputs:
        return_attention=True:
            (features, attention_weights)

        return_sequence=True:
            (features, token_features)

        Both:
            (features, token_features, attention_weights)
    """

    def __init__(
        self,
        vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        encoder_type = encoder_type.lower()

        if encoder_type not in {"rnn", "lstm", "gru", "transformer"}:
            raise ValueError(
                "encoder_type must be rnn, lstm, gru, or transformer"
            )

        if vocab_size <= PAD_IDX:
            raise ValueError("vocab_size must be greater than PAD_IDX")

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        if encoder_type == "transformer" and bidirectional:
            raise ValueError(
                "bidirectional is not applicable to Transformer"
            )

        if encoder_type == "transformer" and hidden_dim % 8 != 0:
            raise ValueError(
                "hidden_dim must be divisible by 8 for Transformer"
            )

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="token_embedding",
        )

        self.embedding_dropout = tf.keras.layers.Dropout(dropout)

        self.rnn_layers = []
        self.inter_layer_dropouts = []
        self.transformer_blocks = []

        self.input_projection = None
        self.positional_encoding = None

        if encoder_type in {"rnn", "lstm", "gru"}:
            self.encoder_output_dim = hidden_dim * self.num_directions

            if encoder_type == "rnn":
                rnn_class = tf.keras.layers.SimpleRNN
            elif encoder_type == "lstm":
                rnn_class = tf.keras.layers.LSTM
            else:
                rnn_class = tf.keras.layers.GRU

            for layer_index in range(num_layers):
                recurrent_layer = rnn_class(
                    units=hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=dropout,
                    name=f"{encoder_type}_layer_{layer_index + 1}",
                )

                if bidirectional:
                    recurrent_layer = tf.keras.layers.Bidirectional(
                        recurrent_layer,
                        merge_mode="concat",
                        name=f"bidirectional_{encoder_type}_{layer_index + 1}",
                    )

                self.rnn_layers.append(recurrent_layer)

                if layer_index < num_layers - 1:
                    self.inter_layer_dropouts.append(
                        tf.keras.layers.Dropout(dropout)
                    )

        else:
            self.encoder_output_dim = hidden_dim

            self.input_projection = tf.keras.layers.Dense(
                hidden_dim,
                name="transformer_input_projection",
            )

            self.positional_encoding = PositionalEncoding(
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

            for layer_index in range(num_layers):
                self.transformer_blocks.append(
                    TransformerEncoderBlock(
                        hidden_dim=hidden_dim,
                        num_heads=8,
                        dropout=dropout,
                        name=f"transformer_block_{layer_index + 1}",
                    )
                )

        if use_attention:
            self.attention = AdditiveAttention(
                hidden_dim=self.encoder_output_dim
            )

            # The attention layer receives the explicit valid-token mask.
            # This also prevents Keras from warning that the incoming
            # embedding mask is being discarded.
            self.attention.supports_masking = True
        else:
            self.attention = None

        self.output_dropout = tf.keras.layers.Dropout(dropout)
        self.output_dim = self.encoder_output_dim

    def call(
        self,
        captions,
        training=None,
        return_attention=False,
        return_sequence=False,
    ):
        captions = tf.convert_to_tensor(captions)

        if captions.shape.rank is not None and captions.shape.rank != 2:
            raise ValueError(
                "Captions must have shape [batch, sequence_length]"
            )

        tf.debugging.assert_rank(
            captions,
            2,
            message="Captions must have shape [batch, sequence_length]",
        )

        valid_mask = tf.not_equal(captions, PAD_IDX)

        lengths = tf.reduce_sum(
            tf.cast(valid_mask, tf.int32),
            axis=1,
        )

        tf.debugging.assert_positive(
            lengths,
            message="Every caption must contain at least one token",
        )

        embedded = self.embedding(captions)
        embedded = self.embedding_dropout(
            embedded,
            training=training,
        )

        if self.encoder_type == "transformer":
            outputs = self.input_projection(embedded)

            outputs = self.positional_encoding(
                outputs,
                training=training,
            )

            for transformer_block in self.transformer_blocks:
                outputs = transformer_block(
                    outputs,
                    valid_mask=valid_mask,
                    training=training,
                )

            hidden = None

        else:
            outputs = embedded
            final_states = None

            for layer_index, recurrent_layer in enumerate(self.rnn_layers):
                recurrent_result = recurrent_layer(
                    outputs,
                    mask=valid_mask,
                    training=training,
                )

                outputs = recurrent_result[0]
                final_states = recurrent_result[1:]

                if layer_index < len(self.inter_layer_dropouts):
                    outputs = self.inter_layer_dropouts[layer_index](
                        outputs,
                        training=training,
                    )

            if self.encoder_type == "lstm":
                if self.bidirectional:
                    forward_hidden = final_states[0]
                    backward_hidden = final_states[2]
                else:
                    forward_hidden = final_states[0]
                    backward_hidden = None
            else:
                if self.bidirectional:
                    forward_hidden = final_states[0]
                    backward_hidden = final_states[1]
                else:
                    forward_hidden = final_states[0]
                    backward_hidden = None

            if self.bidirectional:
                hidden = tf.concat(
                    [forward_hidden, backward_hidden],
                    axis=-1,
                )
            else:
                hidden = forward_hidden

        # Explicitly zero padded token outputs for every architecture.
        outputs = tf.where(
            valid_mask[:, :, tf.newaxis],
            outputs,
            tf.zeros_like(outputs),
        )

        if self.use_attention:
            features, attention_weights = self.attention(
                outputs,
                mask=valid_mask,
            )

        else:
            attention_weights = None

            if self.encoder_type == "transformer":
                mask_float = tf.cast(
                    valid_mask[:, :, tf.newaxis],
                    outputs.dtype,
                )

                summed_outputs = tf.reduce_sum(
                    outputs * mask_float,
                    axis=1,
                )

                valid_counts = tf.reduce_sum(
                    mask_float,
                    axis=1,
                )

                features = summed_outputs / tf.maximum(
                    valid_counts,
                    tf.cast(1.0, outputs.dtype),
                )

            else:
                features = hidden

        features = self.output_dropout(
            features,
            training=training,
        )

        if return_sequence:
            if return_attention:
                return features, outputs, attention_weights

            return features, outputs

        if return_attention:
            return features, attention_weights

        return features


def _run_text_encoder_tests():
    tf.random.set_seed(42)

    vocab_size = 1000
    batch_size = 5
    sequence_length = 34

    dummy_captions = tf.random.uniform(
        shape=(batch_size, sequence_length),
        minval=4,
        maxval=vocab_size,
        dtype=tf.int32,
    )

    lengths = [34, 25, 30, 20, 28]

    caption_array = dummy_captions.numpy()

    for index, length in enumerate(lengths):
        caption_array[index, length:] = PAD_IDX

    dummy_captions = tf.convert_to_tensor(
        caption_array,
        dtype=tf.int32,
    )

    configurations = [
        ("rnn", False, False),
        ("lstm", False, False),
        ("gru", False, False),
        ("rnn", False, True),
        ("lstm", False, True),
        ("gru", False, True),
        ("rnn", True, True),
        ("lstm", True, True),
        ("gru", True, True),
        ("transformer", False, False),
        ("transformer", False, True),
    ]

    print("Text Encoder Architecture Tests")
    print("-" * 65)
    print(f"TensorFlow: {tf.__version__}")
    print(f"Vocabulary size: {vocab_size}")

    for encoder_type, bidirectional, use_attention in configurations:
        model = TextEncoder(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            embedding_dim=300,
            hidden_dim=512,
            num_layers=1,
            dropout=0.2,
            bidirectional=bidirectional,
            use_attention=use_attention,
        )

        features, weights = model(
            dummy_captions,
            training=False,
            return_attention=True,
        )

        expected_dim = 512 * (2 if bidirectional else 1)

        assert tuple(features.shape) == (batch_size, expected_dim)
        assert bool(tf.reduce_all(tf.math.is_finite(features)))

        if use_attention:
            assert tuple(weights.shape) == (batch_size, sequence_length)

            tf.debugging.assert_near(
                tf.reduce_sum(weights, axis=1),
                tf.ones((batch_size,)),
                atol=1e-5,
            )

            padded_weights = tf.boolean_mask(
                weights,
                tf.logical_not(dummy_captions != PAD_IDX),
            )

            tf.debugging.assert_near(
                padded_weights,
                tf.zeros_like(padded_weights),
                atol=1e-6,
            )
        else:
            assert weights is None

        sequence_result = model(
            dummy_captions,
            training=False,
            return_attention=True,
            return_sequence=True,
        )

        sequence_features, token_features, sequence_weights = sequence_result

        assert tuple(sequence_features.shape) == (
            batch_size,
            expected_dim,
        )

        assert tuple(token_features.shape) == (
            batch_size,
            sequence_length,
            expected_dim,
        )

        assert bool(tf.reduce_all(tf.math.is_finite(token_features)))

        padded_token_features = tf.boolean_mask(
            token_features,
            tf.logical_not(dummy_captions != PAD_IDX),
        )

        tf.debugging.assert_near(
            padded_token_features,
            tf.zeros_like(padded_token_features),
            atol=1e-6,
        )

        if use_attention:
            assert tuple(sequence_weights.shape) == (
                batch_size,
                sequence_length,
            )
        else:
            assert sequence_weights is None

        print(
            f"{encoder_type.upper():11} | "
            f"Bi={str(bidirectional):5} | "
            f"Attention={str(use_attention):5} | "
            f"Output={tuple(features.shape)} | "
            f"Sequence={tuple(token_features.shape)} | PASS"
        )

    # Verify gradient flow through a representative LSTM encoder.
    gradient_model = TextEncoder(
        vocab_size=vocab_size,
        encoder_type="lstm",
        embedding_dim=64,
        hidden_dim=128,
        dropout=0.1,
        use_attention=True,
    )

    with tf.GradientTape() as tape:
        gradient_features = gradient_model(
            dummy_captions,
            training=True,
        )
        loss = tf.reduce_mean(tf.square(gradient_features))

    gradients = tape.gradient(
        loss,
        gradient_model.trainable_variables,
    )

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    print("Gradient flow: PASS")

    # Every caption must contain at least one non-padding token.
    invalid_captions = tf.zeros((2, 10), dtype=tf.int32)

    try:
        gradient_model(invalid_captions, training=False)
        raise AssertionError("All-padding captions should have been rejected")
    except tf.errors.InvalidArgumentError:
        print("All-padding input rejection: PASS")

    print("-" * 65)
    print("All Text Encoder tests passed!")


if __name__ == "__main__":
    _run_text_encoder_tests()
