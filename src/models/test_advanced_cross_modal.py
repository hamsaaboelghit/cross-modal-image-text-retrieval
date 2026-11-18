
import pytest
import tensorflow as tf

from src.models.full_model import CrossModalRetrievalModel
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.matching_loss import MatchingBinaryCrossEntropyLoss
from src.evaluation.matching_evaluator import MatchingEvaluator


@pytest.fixture(scope="module")
def images():
    tf.random.set_seed(42)

    return tf.random.uniform(
        (2, 224, 224, 3),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )


@pytest.fixture(scope="module")
def captions():
    tf.random.set_seed(42)

    values = tf.random.uniform(
        (2, 5, 34),
        minval=4,
        maxval=1000,
        dtype=tf.int32,
    )

    return tf.concat(
        [
            values[:, :, :25],
            tf.zeros((2, 5, 9), dtype=tf.int32),
        ],
        axis=-1,
    )


@pytest.fixture(scope="module")
def model(images, captions):
    retrieval_model = CrossModalRetrievalModel(
        vocab_size=1000,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        pretrained=False,
        use_cross_modal_attention=True,
        cross_modal_num_heads=8,
        cross_modal_dropout=0.1,
    )

    retrieval_model(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    return retrieval_model


def test_advanced_model_forward(model, images, captions):
    outputs = model(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    assert tuple(outputs["image_embeddings"].shape) == (2, 512)
    assert tuple(outputs["text_embeddings"].shape) == (2, 5, 512)
    assert tuple(outputs["matching_scores"].shape) == (2, 5)
    assert tuple(
        outputs["cross_modal_attention_weights"].shape
    ) == (2, 5, 34, 49)

    for key in (
        "image_embeddings",
        "text_embeddings",
        "matching_scores",
        "cross_modal_attention_weights",
    ):
        tf.debugging.assert_all_finite(
            outputs[key],
            f"{key} contains NaN or Inf.",
        )


def test_matching_loss_gradient_flow(model, images, captions):
    retrieval_loss_fn = MultiPositiveInfoNCELoss(
        temperature=0.07,
    )
    matching_loss_fn = MatchingBinaryCrossEntropyLoss()

    negative_images = tf.roll(
        images,
        shift=1,
        axis=0,
    )

    with tf.GradientTape() as tape:
        positive_outputs = model(
            images,
            captions,
            training=True,
        )

        negative_outputs = model(
            negative_images,
            captions,
            training=True,
        )

        retrieval_loss = retrieval_loss_fn(
            positive_outputs["image_embeddings"],
            positive_outputs["text_embeddings"],
        )

        matching_loss = matching_loss_fn(
            positive_outputs["matching_scores"],
            negative_outputs["matching_scores"],
        )

        total_loss = retrieval_loss + 0.2 * matching_loss

    gradients = tape.gradient(
        total_loss,
        model.trainable_variables,
    )

    assert gradients
    connected_gradients = [
        gradient for gradient in gradients
        if gradient is not None
    ]
    assert connected_gradients

    for gradient in connected_gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "A model gradient contains NaN or Inf.",
        )

    with tf.GradientTape() as matching_tape:
        positive_outputs = model(
            images,
            captions,
            training=True,
        )
        negative_outputs = model(
            negative_images,
            captions,
            training=True,
        )

        matching_only_loss = matching_loss_fn(
            positive_outputs["matching_scores"],
            negative_outputs["matching_scores"],
        )

    matching_gradients = matching_tape.gradient(
        matching_only_loss,
        model.matching_head.trainable_variables,
    )

    assert matching_gradients
    assert all(g is not None for g in matching_gradients)

    for gradient in matching_gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Matching head gradient contains NaN or Inf.",
        )

    with tf.GradientTape() as attention_tape:
        positive_outputs = model(
            images,
            captions,
            training=True,
        )
        negative_outputs = model(
            negative_images,
            captions,
            training=True,
        )

        attention_loss = matching_loss_fn(
            positive_outputs["matching_scores"],
            negative_outputs["matching_scores"],
        )

    attention_gradients = attention_tape.gradient(
        attention_loss,
        model.cross_modal_attention.trainable_variables,
    )

    assert attention_gradients
    assert all(g is not None for g in attention_gradients)

    for gradient in attention_gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Cross-modal attention gradient contains NaN or Inf.",
        )

    tf.debugging.assert_all_finite(retrieval_loss, "Invalid retrieval loss.")
    tf.debugging.assert_all_finite(matching_loss, "Invalid matching loss.")
    tf.debugging.assert_all_finite(total_loss, "Invalid total loss.")


def test_matching_evaluator(model, images, captions):
    evaluator = MatchingEvaluator(
        model=model,
        threshold=0.0,
    )

    dataset = [
        {
            "image": images,
            "captions": captions,
        }
    ]

    metrics = evaluator.evaluate(dataset)

    required_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "matching_loss",
        "positive_mean_score",
        "negative_mean_score",
        "num_positive_pairs",
        "num_negative_pairs",
    }

    assert required_metrics.issubset(metrics.keys())

    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ):
        assert 0.0 <= metrics[metric_name] <= 1.0

    assert metrics["matching_loss"] >= 0.0
    assert metrics["num_positive_pairs"] == 10
    assert metrics["num_negative_pairs"] == 10
