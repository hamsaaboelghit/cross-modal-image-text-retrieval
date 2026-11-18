
import pytest
import tensorflow as tf

from src.models.baseline_model import (
    BaselineImageEncoder,
    BaselineTextEncoder,
    BaselineRetrievalModel,
)


@pytest.fixture
def sample_inputs():
    tf.random.set_seed(42)

    images = tf.random.uniform(
        (2, 96, 96, 3),
        minval=0,
        maxval=256,
        dtype=tf.float32,
    )

    captions = tf.random.uniform(
        (2, 5, 12),
        minval=4,
        maxval=1000,
        dtype=tf.int32,
    )

    captions = tf.tensor_scatter_nd_update(
        captions,
        indices=[[0, 0, 8], [0, 0, 9], [1, 2, 6], [1, 2, 7]],
        updates=[0, 0, 0, 0],
    )

    return images, captions


def test_baseline_image_encoder(sample_inputs):
    images, _ = sample_inputs

    encoder = BaselineImageEncoder(output_dim=512)
    embeddings = encoder(images, training=False)

    assert tuple(embeddings.shape) == (2, 512)

    tf.debugging.assert_all_finite(embeddings, "Invalid image embeddings")

    tf.debugging.assert_near(
        tf.norm(embeddings, axis=-1),
        tf.ones((2,)),
        atol=1e-5,
    )


def test_baseline_text_encoder_padding_invariance():
    encoder = BaselineTextEncoder(
        vocab_size=1000,
        embedding_dim=64,
        output_dim=512,
        dropout=0.0,
    )

    short_caption = tf.constant([[4, 5, 6, 7, 0, 0]], dtype=tf.int32)
    longer_padding = tf.constant(
        [[4, 5, 6, 7, 0, 0, 0, 0, 0]],
        dtype=tf.int32,
    )

    short_embedding = encoder(short_caption, training=False)
    long_embedding = encoder(longer_padding, training=False)

    tf.debugging.assert_near(
        short_embedding,
        long_embedding,
        atol=1e-5,
    )


def test_baseline_text_encoder_rejects_all_padding():
    encoder = BaselineTextEncoder(vocab_size=1000)

    all_padding = tf.zeros((2, 5), dtype=tf.int32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        encoder(all_padding, training=False)


def test_baseline_retrieval_model(sample_inputs):
    images, captions = sample_inputs

    model = BaselineRetrievalModel(
        vocab_size=1000,
        embedding_dim=64,
        output_dim=512,
        dropout=0.1,
    )

    outputs = model(images, captions, training=False)

    image_embeddings = outputs["image_embeddings"]
    text_embeddings = outputs["text_embeddings"]

    assert tuple(image_embeddings.shape) == (2, 512)
    assert tuple(text_embeddings.shape) == (2, 5, 512)

    tf.debugging.assert_all_finite(image_embeddings, "Invalid image embeddings")
    tf.debugging.assert_all_finite(text_embeddings, "Invalid text embeddings")

    tf.debugging.assert_near(
        tf.norm(image_embeddings, axis=-1),
        tf.ones((2,)),
        atol=1e-5,
    )

    tf.debugging.assert_near(
        tf.norm(text_embeddings, axis=-1),
        tf.ones((2, 5)),
        atol=1e-5,
    )


def test_baseline_model_gradient_flow(sample_inputs):
    images, captions = sample_inputs

    model = BaselineRetrievalModel(
        vocab_size=1000,
        embedding_dim=64,
        output_dim=512,
    )

    with tf.GradientTape() as tape:
        outputs = model(images, captions, training=True)

        loss = (
            tf.reduce_mean(outputs["image_embeddings"])
            + tf.reduce_mean(outputs["text_embeddings"])
        )

    gradients = tape.gradient(loss, model.trainable_variables)

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Non-finite gradient detected",
        )
