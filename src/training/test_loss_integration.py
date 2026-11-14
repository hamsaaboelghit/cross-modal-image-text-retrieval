
import torch

from src.training import (
    MultiPositiveInfoNCELoss,
    MarginBasedContrastiveLoss,
    TripletContrastiveLoss,
)


def test_loss_integration():

    print("=" * 60)
    print("LOSS FUNCTIONS INTEGRATION TEST")
    print("=" * 60)

    # ---------------------------------
    # 1. Device and configuration
    # ---------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 8
    num_captions = 5
    embedding_dim = 512

    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Captions: {num_captions}")
    print(f"Embedding Dimension: {embedding_dim}")

    # ---------------------------------
    # 2. Create shared embeddings
    # ---------------------------------

    torch.manual_seed(42)

    image_embeddings = torch.randn(
        batch_size,
        embedding_dim,
        device=device,
        requires_grad=True
    )

    text_embeddings = torch.randn(
        batch_size,
        num_captions,
        embedding_dim,
        device=device,
        requires_grad=True
    )

    print("\nShared Embeddings:")
    print(f"Images: {image_embeddings.shape}")
    print(f"Texts: {text_embeddings.shape}")

    # ---------------------------------
    # 3. Initialize all loss functions
    # ---------------------------------

    loss_functions = {
        "Multi-Positive InfoNCE": MultiPositiveInfoNCELoss(
            temperature=0.07
        ),
        "Margin-Based Contrastive": MarginBasedContrastiveLoss(
            margin=1.0
        ),
        "Triplet Contrastive": TripletContrastiveLoss(
            margin=0.2
        ),
    }

    # ---------------------------------
    # 4. Test each loss independently
    # ---------------------------------

    results = {}

    for name, criterion in loss_functions.items():

        print("\n" + "-" * 60)
        print(f"Testing: {name}")
        print("-" * 60)

        # Reset gradients before each test
        if image_embeddings.grad is not None:
            image_embeddings.grad = None

        if text_embeddings.grad is not None:
            text_embeddings.grad = None

        # Forward pass
        loss = criterion(
            image_embeddings,
            text_embeddings
        )

        assert torch.is_tensor(loss)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        print(f"Forward Pass: PASS")
        print(f"Loss Value: {loss.item():.6f}")

        # Backward pass
        loss.backward()

        assert image_embeddings.grad is not None
        assert text_embeddings.grad is not None

        assert torch.isfinite(
            image_embeddings.grad
        ).all().item()

        assert torch.isfinite(
            text_embeddings.grad
        ).all().item()

        print("Image Gradients: PASS")
        print("Text Gradients: PASS")
        print("Backward Pass: PASS")

        results[name] = loss.item()

    # ---------------------------------
    # 5. Final integration summary
    # ---------------------------------

    print("\n" + "=" * 60)
    print("INTEGRATION RESULTS")
    print("=" * 60)

    for name, value in results.items():
        print(f"{name}: {value:.6f} | PASS")

    assert len(results) == 3

    print("\nAll 3 Loss Functions integrated successfully!")
    print("All Integration Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_loss_integration()
