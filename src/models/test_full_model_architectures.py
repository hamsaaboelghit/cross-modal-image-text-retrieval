
import pytest
import tensorflow as tf

from src.models.text_encoder import TextEncoder


CONFIGURATIONS = [
    ("rnn", False),
    ("lstm", False),
    ("gru", False),
    ("rnn", True),
    ("lstm", True),
    ("gru", True),
    ("transformer", False),
]


@pytest.fixture
def captions():
    tf.random.set_seed(42)

    values = tf.random.uniform(
        (2, 12),
        minval=4,
        maxval=1000,
        dtype=tf.int32,
    )

    values = values.numpy()
    values[0, 8:] = 0
    values[1, 10:] = 0

    return tf.convert_to_tensor(values)


@pytest.mark.parametrize(
    "encoder_type,bidirectional",
    CONFIGURATIONS,
)
def test_encoder_architecture(encoder_type, bidirectional, captions):
    encoder = TextEncoder(
        vocab_size=1000,
        encoder_type=encoder_type,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=bidirectional,
        use_attention=True,
    )

    features, attention = encoder(
        captions,
        training=False,
        return_attention=True,
    )

    expected_dim = 64 * (2 if bidirectional else 1)

    assert tuple(features.shape) == (2, expected_dim)
    assert tuple(attention.shape) == (2, 12)

    tf.debugging.assert_all_finite(features, "Invalid encoder features")
    tf.debugging.assert_all_finite(attention, "Invalid attention weights")

    tf.debugging.assert_near(
        tf.reduce_sum(attention, axis=1),
        tf.ones((2,)),
        atol=1e-4,
    )

    padding_mask = tf.equal(captions, 0)
    padded_attention = tf.boolean_mask(attention, padding_mask)

    tf.debugging.assert_near(
        padded_attention,
        tf.zeros_like(padded_attention),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "encoder_type,bidirectional",
    CONFIGURATIONS,
)
def test_encoder_gradient_flow(encoder_type, bidirectional, captions):
    encoder = TextEncoder(
        vocab_size=1000,
        encoder_type=encoder_type,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=bidirectional,
        use_attention=True,
    )

    with tf.GradientTape() as tape:
        features = encoder(captions, training=True)
        loss = tf.reduce_mean(tf.square(features))

    gradients = tape.gradient(loss, encoder.trainable_variables)

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Non-finite gradient detected",
        )


def test_transformer_rejects_bidirectional():
    with pytest.raises(ValueError):
        TextEncoder(
            vocab_size=1000,
            encoder_type="transformer",
            hidden_dim=64,
            bidirectional=True,
        )


def test_encoder_rejects_all_padding():
    encoder = TextEncoder(
        vocab_size=1000,
        encoder_type="lstm",
        hidden_dim=64,
        use_attention=True,
    )

    invalid_captions = tf.zeros((2, 10), dtype=tf.int32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        encoder(invalid_captions, training=False)
