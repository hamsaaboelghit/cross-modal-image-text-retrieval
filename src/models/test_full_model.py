
import pytest
import tensorflow as tf

from src.models.full_model import CrossModalRetrievalModel


@pytest.fixture
def sample_inputs():
    tf.random.set_seed(42)

    images = tf.random.uniform(
        (2, 224, 224, 3),
        minval=0,
        maxval=256,
        dtype=tf.float32,
    )

    captions = tf.random.uniform(
        (2, 3, 12),
        minval=4,
        maxval=1000,
        dtype=tf.int32,
    )

    captions_array = captions.numpy()
    captions_array[0, 1, 8:] = 0
    captions_array[1, 2, 6:] = 0

    return images, tf.convert_to_tensor(captions_array)


def create_model():
    return CrossModalRetrievalModel(
        vocab_size=1000,
        encoder_type="gru",
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        use_attention=True,
        pretrained=False,
        use_cross_modal_attention=False,
    )


def test_full_model_forward_pass(sample_inputs):
    images, captions = sample_inputs
    model = create_model()

    outputs = model(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    assert tuple(outputs["image_embeddings"].shape) == (2, 512)
    assert tuple(outputs["text_embeddings"].shape) == (2, 3, 512)
    assert tuple(outputs["attention_weights"].shape) == (2, 3, 12)

    tf.debugging.assert_all_finite(
        outputs["image_embeddings"],
        "Invalid image embeddings",
    )

    tf.debugging.assert_all_finite(
        outputs["text_embeddings"],
        "Invalid text embeddings",
    )

    tf.debugging.assert_all_finite(
        outputs["attention_weights"],
        "Invalid attention weights",
    )


def test_embedding_normalization(sample_inputs):
    images, captions = sample_inputs
    outputs = create_model()(
        images,
        captions,
        training=False,
    )

    tf.debugging.assert_near(
        tf.norm(outputs["image_embeddings"], axis=-1),
        tf.ones((2,)),
        atol=1e-4,
    )

    tf.debugging.assert_near(
        tf.norm(outputs["text_embeddings"], axis=-1),
        tf.ones((2, 3)),
        atol=1e-4,
    )


def test_attention_ignores_padding(sample_inputs):
    images, captions = sample_inputs

    outputs = create_model()(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    flat_captions = tf.reshape(captions, (-1, 12))
    flat_attention = tf.reshape(outputs["attention_weights"], (-1, 12))

    padding_mask = tf.equal(flat_captions, 0)
    padded_weights = tf.boolean_mask(flat_attention, padding_mask)

    tf.debugging.assert_near(
        padded_weights,
        tf.zeros_like(padded_weights),
        atol=1e-6,
    )


def test_independent_encoders(sample_inputs):
    images, captions = sample_inputs
    model = create_model()

    image_embeddings = model.encode_image(images, training=False)

    flat_captions = tf.reshape(captions, (-1, 12))
    text_embeddings = model.encode_text(flat_captions, training=False)

    assert tuple(image_embeddings.shape) == (2, 512)
    assert tuple(text_embeddings.shape) == (6, 512)


def test_full_model_gradient_flow(sample_inputs):
    images, captions = sample_inputs
    model = create_model()

    with tf.GradientTape() as tape:
        outputs = model(images, captions, training=True)

        loss = (
            1.0
            - tf.reduce_mean(
                tf.reduce_sum(
                    outputs["image_embeddings"][:, tf.newaxis, :]
                    * outputs["text_embeddings"],
                    axis=-1,
                )
            )
        )

    gradients = tape.gradient(loss, model.trainable_variables)

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Non-finite gradient detected",
        )
