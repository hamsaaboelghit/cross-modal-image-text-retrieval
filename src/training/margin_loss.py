
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginBasedContrastiveLoss(nn.Module):
    """
    Pairwise Contrastive Loss for image-text embeddings.

    Positive pairs:
        Image and its own captions.

    Negative pairs:
        Image and captions belonging to other images.
    """

    def __init__(self, margin=1.0):
        super().__init__()

        if margin <= 0:
            raise ValueError("Margin must be positive.")

        self.margin = margin

    def forward(self, image_embeddings, text_embeddings):
        """
        image_embeddings: [B, D]
        text_embeddings:  [B, C, D]

        B = batch size
        C = captions per image
        D = embedding dimension
        """

        batch_size = image_embeddings.size(0)

        if batch_size < 2:
            raise ValueError(
                "Batch size must be at least 2 for negative pairs."
            )

        if text_embeddings.size(0) != batch_size:
            raise ValueError(
                "Image and text batch sizes must match."
            )

        # Normalize embeddings
        image_embeddings = F.normalize(
            image_embeddings, p=2, dim=-1
        )

        text_embeddings = F.normalize(
            text_embeddings, p=2, dim=-1
        )

        # Flatten captions: [B, C, D] -> [B*C, D]
        num_captions = text_embeddings.size(1)

        text_flat = text_embeddings.reshape(
            batch_size * num_captions, -1
        )

        # Calculate Euclidean distances
        distances = torch.cdist(
            image_embeddings,
            text_flat,
            p=2
        )

        # Caption ownership:
        # [0,0,0,0,0, 1,1,1,1,1, ...]
        caption_image_ids = torch.arange(
            batch_size, device=image_embeddings.device
        ).repeat_interleave(num_captions)

        image_ids = torch.arange(
            batch_size, device=image_embeddings.device
        )

        # Positive mask: captions belonging to each image
        positive_mask = (
            image_ids.unsqueeze(1)
            == caption_image_ids.unsqueeze(0)
        )

        negative_mask = ~positive_mask

        # Image-to-Text loss
        positive_distances = distances[positive_mask]
        negative_distances = distances[negative_mask]

        positive_loss_i2t = positive_distances.pow(2).mean()

        negative_loss_i2t = F.relu(
            self.margin - negative_distances
        ).pow(2).mean()

        loss_i2t = (
            positive_loss_i2t + negative_loss_i2t
        ) / 2

        # Text-to-Image loss
        distances_t2i = distances.T
        positive_mask_t2i = positive_mask.T
        negative_mask_t2i = negative_mask.T

        positive_distances_t2i = distances_t2i[
            positive_mask_t2i
        ]

        negative_distances_t2i = distances_t2i[
            negative_mask_t2i
        ]

        positive_loss_t2i = (
            positive_distances_t2i.pow(2).mean()
        )

        negative_loss_t2i = F.relu(
            self.margin - negative_distances_t2i
        ).pow(2).mean()

        loss_t2i = (
            positive_loss_t2i + negative_loss_t2i
        ) / 2

        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2

        return loss


def test_margin_loss():

    print("Margin-Based Contrastive Loss Tests")
    print("-" * 45)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    batch_size = 8
    num_captions = 5
    embedding_dim = 512

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

    print(f"Images: {image_embeddings.shape}")
    print(f"Captions: {text_embeddings.shape}")

    criterion = MarginBasedContrastiveLoss(
        margin=1.5
    ).to(device)

    loss = criterion(
        image_embeddings,
        text_embeddings
    )

    print(f"Loss: {loss.item():.4f}")

    # Check loss
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0

    # Backpropagation
    loss.backward()

    assert image_embeddings.grad is not None
    assert text_embeddings.grad is not None

    assert torch.isfinite(
        image_embeddings.grad
    ).all()

    assert torch.isfinite(
        text_embeddings.grad
    ).all()

    print("\nGradient checks passed!")

    # Check invalid batch size
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

    print("\nAll Margin-Based Contrastive Loss tests passed!")


if __name__ == "__main__":
    test_margin_loss()
