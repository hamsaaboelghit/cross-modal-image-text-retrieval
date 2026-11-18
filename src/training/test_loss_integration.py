
import tensorflow as tf

from src.training import (
    MultiPositiveInfoNCELoss,
    MarginBasedContrastiveLoss,
    TripletContrastiveLoss,
)


def test_loss_integration():

    print("\n" + "=" * 60)
    print("VISIONMATCH AI - LOSS FUNCTIONS INTEGRATION TEST")
    print("=" * 60)

    tf.keras.utils.set_random_seed(42)

    batch_size = 8
    num_captions = 5
    embedding_dim = 512

    print(f"TensorFlow: {tf.__version__}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Captions: {num_captions}")
    print(f"Embedding Dimension: {embedding_dim}")

    # -------------------------------------------------
    # 1. Create shared embeddings
    # -------------------------------------------------

    image_embeddings = tf.Variable(
        tf.random.normal(
            (batch_size, embedding_dim)
        )
    )

    text_embeddings = tf.Variable(
        tf.random.normal(
            (
                batch_size,
                num_captions,
                embedding_dim,
            )
        )
    )

    print("\nShared Embeddings:")
    print(
        f"Images: {image_embeddings.shape}"
    )
    print(
        f"Texts: {text_embeddings.shape}"
    )

    # -------------------------------------------------
    # 2. Initialize loss functions
    # -------------------------------------------------

    loss_functions = {
        "Multi-Positive InfoNCE": (
            MultiPositiveInfoNCELoss(
                temperature=0.07
            )
        ),

        "Margin-Based Contrastive": (
            MarginBasedContrastiveLoss(
                margin=1.0
            )
        ),

        "Triplet Contrastive": (
            TripletContrastiveLoss(
                margin=0.2
            )
        ),
    }

    results = {}

    # -------------------------------------------------
    # 3. Test every loss independently
    # -------------------------------------------------

    for name, criterion in loss_functions.items():

        print("\n" + "-" * 60)
        print(f"Testing: {name}")
        print("-" * 60)

        with tf.GradientTape() as tape:

            loss = criterion(
                image_embeddings,
                text_embeddings,
            )

        # Forward pass validation.

        assert loss.shape.rank == 0

        tf.debugging.assert_all_finite(
            loss,
            f"{name}: Loss contains NaN or Inf.",
        )

        tf.debugging.assert_greater_equal(
            loss,
            0.0,
            message=f"{name}: Loss must be non-negative.",
        )

        print("Forward Pass: PASS")
        print(
            f"Loss Value: {float(loss.numpy()):.6f}"
        )

        # Backward pass using GradientTape.

        gradients = tape.gradient(
            loss,
            [
                image_embeddings,
                text_embeddings,
            ],
        )

        image_gradient, text_gradient = gradients

        assert image_gradient is not None, (
            f"{name}: Image gradient is missing."
        )

        assert text_gradient is not None, (
            f"{name}: Text gradient is missing."
        )

        tf.debugging.assert_all_finite(
            image_gradient,
            f"{name}: Image gradient is not finite.",
        )

        tf.debugging.assert_all_finite(
            text_gradient,
            f"{name}: Text gradient is not finite.",
        )

        print("Image Gradients: PASS")
        print("Text Gradients: PASS")
        print("Gradient Flow: PASS")

        results[name] = float(loss.numpy())

    # -------------------------------------------------
    # 4. Additional InfoNCE validation
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("Additional InfoNCE Validation")
    print("-" * 60)

    info_nce = MultiPositiveInfoNCELoss(
        temperature=0.07
    )

    perfect_images = tf.eye(
        4,
        dtype=tf.float32,
    )

    perfect_texts = tf.repeat(
        perfect_images[:, tf.newaxis, :],
        repeats=2,
        axis=1,
    )

    perfect_loss = info_nce(
        perfect_images,
        perfect_texts,
    )

    random_images = tf.random.normal(
        (4, 4)
    )

    random_texts = tf.random.normal(
        (4, 2, 4)
    )

    random_loss = info_nce(
        random_images,
        random_texts,
    )

    tf.debugging.assert_all_finite(
        perfect_loss,
        "Perfect-pair loss is not finite.",
    )

    tf.debugging.assert_all_finite(
        random_loss,
        "Random-pair loss is not finite.",
    )

    print("Perfect positive pairs: PASS")
    print("Random pair loss: PASS")

    # -------------------------------------------------
    # 5. Margin loss validation
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("Margin Loss Validation")
    print("-" * 60)

    margin_loss = MarginBasedContrastiveLoss(
        margin=1.0
    )

    masked_caption_embeddings = tf.random.normal(
        (4, 3, 16)
    )

    masked_image_embeddings = tf.random.normal(
        (4, 16)
    )

    caption_mask = tf.constant(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [True, True, False],
        ],
        dtype=tf.bool,
    )

    masked_loss = margin_loss(
        masked_image_embeddings,
        masked_caption_embeddings,
        caption_mask=caption_mask,
    )

    tf.debugging.assert_all_finite(
        masked_loss,
        "Masked margin loss is not finite.",
    )

    tf.debugging.assert_greater_equal(
        masked_loss,
        0.0,
    )

    print("Caption mask validation: PASS")

    # -------------------------------------------------
    # 6. Triplet loss validation
    # -------------------------------------------------

    print("\n" + "-" * 60)
    print("Triplet Loss Validation")
    print("-" * 60)

    triplet_loss = TripletContrastiveLoss(
        margin=0.2
    )

    positive_indices = tf.constant(
        [0, 1, 2, 3, 4, 0, 1, 2],
        dtype=tf.int32,
    )

    indexed_loss = triplet_loss(
        image_embeddings,
        text_embeddings,
        positive_caption_indices=positive_indices,
    )

    tf.debugging.assert_all_finite(
        indexed_loss,
        "Indexed triplet loss is not finite.",
    )

    print("Explicit positive caption indices: PASS")

    # -------------------------------------------------
    # 7. Final summary
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("LOSS INTEGRATION RESULTS")
    print("=" * 60)

    for name, value in results.items():

        print(
            f"{name}: {value:.6f} | PASS"
        )

    assert len(results) == 3

    print("\nAll 3 Loss Functions Passed!")
    print("Forward and Gradient Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_loss_integration()
