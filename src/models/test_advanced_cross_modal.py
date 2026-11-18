
import tensorflow as tf

from src.models.full_model import CrossModalRetrievalModel
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.matching_loss import MatchingBinaryCrossEntropyLoss
from src.evaluation.matching_evaluator import MatchingEvaluator


def test_advanced_model_forward():
    tf.random.set_seed(42)

    batch_size = 2
    num_captions = 5
    sequence_length = 34
    vocab_size = 1000

    images = tf.random.uniform(
        (batch_size, 224, 224, 3),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )

    captions = tf.random.uniform(
        (batch_size, num_captions, sequence_length),
        minval=4,
        maxval=vocab_size,
        dtype=tf.int32,
    )

    # Add padding to test the text mask.
    captions = tf.concat(
        [
            captions[:, :, :25],
            tf.zeros(
                (batch_size, num_captions, 9),
                dtype=tf.int32,
            ),
        ],
        axis=-1,
    )

    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
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

    outputs = model(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    assert tuple(outputs["image_embeddings"].shape) == (
        batch_size,
        512,
    )

    assert tuple(outputs["text_embeddings"].shape) == (
        batch_size,
        num_captions,
        512,
    )

    assert tuple(outputs["matching_scores"].shape) == (
        batch_size,
        num_captions,
    )

    assert tuple(
        outputs["cross_modal_attention_weights"].shape
    ) == (
        batch_size,
        num_captions,
        sequence_length,
        49,
    )

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

    print("Advanced model forward pass: PASS")
    print("Image embeddings shape: PASS")
    print("Text embeddings shape: PASS")
    print("Matching scores shape: PASS")
    print("Cross-modal attention shape: PASS")

    return model, images, captions


def test_matching_loss_gradient_flow(model, images, captions):
    retrieval_loss_fn = MultiPositiveInfoNCELoss(
        temperature=0.07,
    )

    matching_loss_fn = MatchingBinaryCrossEntropyLoss()

    with tf.GradientTape() as tape:
        positive_outputs = model(
            images,
            captions,
            training=True,
        )

        negative_images = tf.roll(
            images,
            shift=1,
            axis=0,
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

    assert len(gradients) > 0

    connected_gradients = [
        gradient
        for gradient in gradients
        if gradient is not None
    ]

    assert connected_gradients, (
        "No gradients were connected to model variables."
    )

    for gradient in connected_gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "A gradient contains NaN or Inf.",
        )

    # Explicitly verify that the matching head receives gradients.
    matching_head_variables = (
        model.matching_head.trainable_variables
    )

    matching_head_gradients = tape.gradient(
        total_loss,
        matching_head_variables,
    ) if False else None

    # Use a second tape to inspect matching-head gradients directly.
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
        matching_head_variables,
    )

    assert matching_gradients
    assert all(
        gradient is not None
        for gradient in matching_gradients
    )

    for gradient in matching_gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Matching head gradient contains NaN or Inf.",
        )

    # Cross-modal attention must also receive matching-loss gradients.
    attention_variables = (
        model.cross_modal_attention.trainable_variables
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
        attention_variables,
    )

    assert attention_gradients
    assert all(
        gradient is not None
        for gradient in attention_gradients
    )

    for gradient in attention_gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Cross-modal attention gradient contains NaN or Inf.",
        )

    tf.debugging.assert_all_finite(
        retrieval_loss,
        "Retrieval loss is not finite.",
    )

    tf.debugging.assert_all_finite(
        matching_loss,
        "Matching loss is not finite.",
    )

    tf.debugging.assert_all_finite(
        total_loss,
        "Total loss is not finite.",
    )

    print(f"Retrieval loss: {retrieval_loss.numpy():.4f}")
    print(f"Matching loss: {matching_loss.numpy():.4f}")
    print(f"Total loss: {total_loss.numpy():.4f}")

    print("Combined loss is finite: PASS")
    print("Model gradient flow: PASS")
    print("Matching head gradient flow: PASS")
    print("Cross-modal attention gradient flow: PASS")


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
        "matching_loss",
    ):
        value = metrics[metric_name]
        assert 0.0 <= value <= 1.0 if metric_name != "matching_loss" else value >= 0.0

    assert metrics["num_positive_pairs"] == 10
    assert metrics["num_negative_pairs"] == 10

    print("Matching evaluator: PASS")
    print("Evaluation metrics:", metrics)


def run_tests():
    print("=" * 65)
    print("ADVANCED CROSS-MODAL ARCHITECTURE TESTS")
    print("=" * 65)

    model, images, captions = test_advanced_model_forward()

    test_matching_loss_gradient_flow(
        model,
        images,
        captions,
    )

    test_matching_evaluator(
        model,
        images,
        captions,
    )

    print("=" * 65)
    print("ALL ADVANCED ARCHITECTURE TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
