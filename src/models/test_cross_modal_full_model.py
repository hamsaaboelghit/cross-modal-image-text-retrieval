
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

    caption_array = captions.numpy()
    caption_array[0, 0, 9:] = 0
    caption_array[1, 2, 7:] = 0

    return images, tf.convert_to_tensor(caption_array)


def create_advanced_model():
    return CrossModalRetrievalModel(
        vocab_size=1000,
        encoder_type="gru",
        embedding_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        pretrained=False,
        use_cross_modal_attention=True,
    )


def test_cross_modal_output_shapes(sample_inputs):
    images, captions = sample_inputs
    model = create_advanced_model()

    outputs = model(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    assert tuple(outputs["image_embeddings"].shape) == (2, 512)
    assert tuple(outputs["text_embeddings"].shape) == (2, 3, 512)
    assert tuple(outputs["matching_scores"].shape) == (2, 3)
    assert tuple(outputs["negative_matching_scores"].shape) == (2, 3)
    assert tuple(outputs["attention_weights"].shape) == (2, 3, 12)
    assert tuple(
        outputs["cross_modal_attention_weights"].shape
    ) == (2, 3, 12, 49)

    assert bool(outputs["matching_negatives_valid"].numpy())


def test_cross_modal_attention_padding_mask(sample_inputs):
    images, captions = sample_inputs

    outputs = create_advanced_model()(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    flat_captions = tf.reshape(captions, (-1, 12))
    padding_mask = tf.equal(flat_captions, 0)

    text_attention = tf.reshape(
        outputs["attention_weights"],
        (-1, 12),
    )

    cross_attention = tf.reshape(
        outputs["cross_modal_attention_weights"],
        (-1, 12, 49),
    )

    padded_text_attention = tf.boolean_mask(
        text_attention,
        padding_mask,
    )

    padded_cross_attention = tf.boolean_mask(
        cross_attention,
        padding_mask,
    )

    tf.debugging.assert_near(
        padded_text_attention,
        tf.zeros_like(padded_text_attention),
        atol=1e-6,
    )

    tf.debugging.assert_near(
        padded_cross_attention,
        tf.zeros_like(padded_cross_attention),
        atol=1e-6,
    )


def test_cross_modal_attention_normalization(sample_inputs):
    images, captions = sample_inputs

    outputs = create_advanced_model()(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    attention = outputs["cross_modal_attention_weights"]
    attention_sums = tf.reduce_sum(attention, axis=-1)
    valid_tokens = tf.not_equal(captions, 0)

    valid_sums = tf.boolean_mask(
        attention_sums,
        valid_tokens,
    )

    tf.debugging.assert_near(
        valid_sums,
        tf.ones_like(valid_sums),
        atol=1e-4,
    )


def test_cross_modal_gradient_flow(sample_inputs):
    images, captions = sample_inputs
    model = create_advanced_model()

    with tf.GradientTape() as tape:
        outputs = model(
            images,
            captions,
            training=True,
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
        )

        positive_loss = loss_fn(
            tf.ones_like(outputs["matching_scores"]),
            outputs["matching_scores"],
        )

        negative_loss = loss_fn(
            tf.zeros_like(outputs["negative_matching_scores"]),
            outputs["negative_matching_scores"],
        )

        # Retrieval loss connects both image and text projection heads.
        image_embeddings = outputs["image_embeddings"]
        text_embeddings = outputs["text_embeddings"]

        retrieval_similarity = tf.reduce_sum(
            image_embeddings[:, tf.newaxis, :]
            * text_embeddings,
            axis=-1,
        )

        retrieval_loss = tf.reduce_mean(
            1.0 - retrieval_similarity
        )

        total_loss = (
            positive_loss
            + negative_loss
            + retrieval_loss
        )

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables,
    )

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Non-finite gradient detected",
        )


def test_single_image_batch_has_no_valid_negatives():
    model = create_advanced_model()

    images = tf.random.uniform(
        (1, 224, 224, 3),
        maxval=256,
    )

    captions = tf.random.uniform(
        (1, 2, 10),
        minval=4,
        maxval=1000,
        dtype=tf.int32,
    )

    outputs = model(images, captions, training=False)

    assert not bool(
        outputs["matching_negatives_valid"].numpy()
    )
