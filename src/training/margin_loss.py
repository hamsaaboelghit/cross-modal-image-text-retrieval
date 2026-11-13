
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginBasedContrastiveLoss(nn.Module):
    """
    Pairwise Contrastive Loss for image-text retrieval.

    Image embeddings:
        [B, D]

    Text embeddings:
        [B, C, D]

    Positive pairs:
        Each image with its own valid captions.

    Negative pairs:
        Each image with captions belonging to other images.

    The embeddings are L2-normalized before calculating
    Euclidean distances.
    """

    def __init__(self, margin=1.0):
        super().__init__()

        if not isinstance(margin, (int, float)):
            raise TypeError("Margin must be a number.")

        if not 0 < margin <= 2:
            raise ValueError(
                "Margin must be greater than 0 and at most 2 "
                "for normalized embeddings."
            )

        self.margin = float(margin)

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        caption_mask=None
    ):
        # ---------------------------------
        # 1. Input validation
        # ---------------------------------

        if not torch.is_tensor(image_embeddings):
            raise TypeError("image_embeddings must be a Tensor.")

        if not torch.is_tensor(text_embeddings):
            raise TypeError("text_embeddings must be a Tensor.")

        if image_embeddings.ndim != 2:
            raise ValueError(
                "image_embeddings must have shape [B, D]."
            )

        if text_embeddings.ndim != 3:
            raise ValueError(
                "text_embeddings must have shape [B, C, D]."
            )

        batch_size, embedding_dim = image_embeddings.shape

        text_batch_size, num_captions, text_dim = (
            text_embeddings.shape
        )

        if batch_size < 2:
            raise ValueError(
                "Batch size must be at least 2."
            )

        if num_captions < 1:
            raise ValueError(
                "At least one caption per image is required."
            )

        if embedding_dim < 1 or text_dim < 1:
            raise ValueError(
                "Embedding dimensions must be positive."
            )

        if batch_size != text_batch_size:
            raise ValueError(
                "Image and text batch sizes must match."
            )

        if embedding_dim != text_dim:
            raise ValueError(
                "Image and text embedding dimensions must match."
            )

        if image_embeddings.device != text_embeddings.device:
            raise ValueError(
                "Image and text embeddings must be on the same device."
            )

        if not image_embeddings.is_floating_point():
            raise TypeError(
                "Image embeddings must have a floating-point dtype."
            )

        if not text_embeddings.is_floating_point():
            raise TypeError(
                "Text embeddings must have a floating-point dtype."
            )

        # ---------------------------------
        # 2. Caption validity mask
        # ---------------------------------

        if caption_mask is None:
            caption_mask = torch.ones(
                batch_size,
                num_captions,
                dtype=torch.bool,
                device=image_embeddings.device
            )

        else:
            if not torch.is_tensor(caption_mask):
                raise TypeError("caption_mask must be a Tensor.")

            if caption_mask.shape != (batch_size, num_captions):
                raise ValueError(
                    "caption_mask must have shape [B, C]."
                )

            caption_mask = caption_mask.to(
                device=image_embeddings.device,
                dtype=torch.bool
            )

        if not caption_mask.any(dim=1).all().item():
            raise ValueError(
                "Every image must have at least one valid caption."
            )

        # ---------------------------------
        # 3. Normalize embeddings
        # ---------------------------------

        image_embeddings = F.normalize(
            image_embeddings.float(),
            p=2,
            dim=-1
        )

        text_embeddings = F.normalize(
            text_embeddings.float(),
            p=2,
            dim=-1
        )

        # ---------------------------------
        # 4. Flatten text embeddings
        # ---------------------------------

        text_flat = text_embeddings.reshape(
            batch_size * num_captions,
            embedding_dim
        )

        valid_captions = caption_mask.reshape(-1)

        # ---------------------------------
        # 5. Calculate pairwise distances
        # ---------------------------------

        distances = torch.cdist(
            image_embeddings,
            text_flat,
            p=2
        )

        # ---------------------------------
        # 6. Build positive and negative masks
        # ---------------------------------

        image_ids = torch.arange(
            batch_size,
            device=image_embeddings.device
        )

        caption_image_ids = torch.arange(
            batch_size,
            device=image_embeddings.device
        ).repeat_interleave(num_captions)

        positive_mask = (
            image_ids.unsqueeze(1)
            == caption_image_ids.unsqueeze(0)
        )

        valid_mask = valid_captions.unsqueeze(0).expand(
            batch_size, -1
        )

        positive_mask = positive_mask & valid_mask

        negative_mask = (~positive_mask) & valid_mask

        # ---------------------------------
        # 7. Positive pair loss
        # ---------------------------------

        positive_distances = distances[positive_mask]

        positive_loss = positive_distances.square().mean()

        # ---------------------------------
        # 8. Negative pair loss
        # ---------------------------------

        negative_distances = distances[negative_mask]

        negative_loss = F.relu(
            self.margin - negative_distances
        ).square().mean()

        # ---------------------------------
        # 9. Final contrastive loss
        # ---------------------------------

        loss = 0.5 * (
            positive_loss + negative_loss
        )

        return loss


def test_margin_loss():

    print("Margin-Based Contrastive Loss Tests")
    print("-" * 50)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    criterion = MarginBasedContrastiveLoss(
        margin=1.0
    ).to(device)

    # ---------------------------------
    # Test 1: Deterministic mathematical test
    # ---------------------------------

    images = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0]
        ],
        device=device
    )

    captions = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]]
        ],
        device=device
    )

    expected_loss = 0.75

    actual_loss = criterion(images, captions)

    assert torch.isclose(
        actual_loss,
        torch.tensor(
            expected_loss,
            device=device
        ),
        atol=1e-5
    ), f"Expected {expected_loss}, got {actual_loss.item()}"

    print("Deterministic mathematical test passed!")

    # ---------------------------------
    # Test 2: Perfectly separated pairs
    # ---------------------------------

    perfect_images = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0]
        ],
        device=device
    )

    perfect_captions = perfect_images.unsqueeze(1).repeat(
        1, 2, 1
    )

    perfect_loss = criterion(
        perfect_images,
        perfect_captions
    )

    assert torch.isclose(
        perfect_loss,
        torch.tensor(0.0, device=device),
        atol=1e-5
    )

    print("Perfect matching test passed!")

    # ---------------------------------
    # Test 3: Caption mask
    # ---------------------------------

    images = torch.randn(
        4, 16, device=device
    )

    captions = torch.randn(
        4, 3, 16, device=device
    )

    caption_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [True, True, False]
        ],
        device=device
    )

    masked_loss = criterion(
        images,
        captions,
        caption_mask=caption_mask
    )

    assert torch.isfinite(masked_loss)
    assert masked_loss.item() >= 0

    print("Caption mask test passed!")

    # ---------------------------------
    # Test 4: Backpropagation
    # ---------------------------------

    image_embeddings = torch.randn(
        8,
        512,
        device=device,
        requires_grad=True
    )

    text_embeddings = torch.randn(
        8,
        5,
        512,
        device=device,
        requires_grad=True
    )

    loss = criterion(
        image_embeddings,
        text_embeddings
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0

    loss.backward()

    assert image_embeddings.grad is not None
    assert text_embeddings.grad is not None

    assert torch.isfinite(
        image_embeddings.grad
    ).all()

    assert torch.isfinite(
        text_embeddings.grad
    ).all()

    print("Backpropagation test passed!")

    # ---------------------------------
    # Test 5: Invalid batch size
    # ---------------------------------

    try:
        criterion(
            image_embeddings[:1],
            text_embeddings[:1]
        )

        raise AssertionError(
            "Expected ValueError for batch size 1."
        )

    except ValueError:
        print("Batch size validation passed!")

    # ---------------------------------
    # Test 6: Invalid margin
    # ---------------------------------

    try:
        MarginBasedContrastiveLoss(margin=2.5)

        raise AssertionError(
            "Expected ValueError for invalid margin."
        )

    except ValueError:
        print("Margin validation passed!")

    print("-" * 50)
    print("All Margin-Based Contrastive Loss tests passed!")


if __name__ == "__main__":
    test_margin_loss()
