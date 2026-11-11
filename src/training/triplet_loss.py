
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletContrastiveLoss(nn.Module):

    def __init__(self, margin=0.2):

        super().__init__()

        if margin <= 0:
            raise ValueError("Margin must be positive")

        self.margin = margin

    def forward(
        self,
        image_embeddings,
        text_embeddings,
        positive_caption_indices=None
    ):

        # image_embeddings: [B, D]
        # text_embeddings: [B, C, D]

        batch_size, num_captions, embedding_dim = (
            text_embeddings.shape
        )

        if batch_size < 2:
            raise ValueError(
                "Triplet loss requires at least 2 images"
            )

        if image_embeddings.shape != (
            batch_size,
            embedding_dim
        ):
            raise ValueError(
                "Image and text embedding dimensions must match"
            )

        if num_captions < 1:
            raise ValueError(
                "Each image must have at least one caption"
            )

        # Normalize embeddings
        image_embeddings = F.normalize(
            image_embeddings,
            p=2,
            dim=-1
        )

        text_embeddings = F.normalize(
            text_embeddings,
            p=2,
            dim=-1
        )

        # Randomly select one positive caption per image
        if positive_caption_indices is None:

            positive_caption_indices = torch.randint(
                low=0,
                high=num_captions,
                size=(batch_size,),
                device=text_embeddings.device
            )

        else:

            positive_caption_indices = (
                positive_caption_indices.to(
                    text_embeddings.device
                )
            )

            if positive_caption_indices.shape != (
                batch_size,
            ):
                raise ValueError(
                    "Positive indices must have shape [B]"
                )

            if torch.any(positive_caption_indices < 0) or torch.any(
                positive_caption_indices >= num_captions
            ):
                raise ValueError(
                    "Positive caption index out of range"
                )

        # Select positive captions
        batch_indices = torch.arange(
            batch_size,
            device=text_embeddings.device
        )

        positive_embeddings = text_embeddings[
            batch_indices,
            positive_caption_indices
        ]

        # Random negative image index, guaranteed different
        offsets = torch.randint(
            low=1,
            high=batch_size,
            size=(batch_size,),
            device=image_embeddings.device
        )

        negative_image_indices = (
            batch_indices + offsets
        ) % batch_size

        # Select a random caption from another image
        negative_caption_indices = torch.randint(
            low=0,
            high=num_captions,
            size=(batch_size,),
            device=text_embeddings.device
        )

        negative_embeddings = text_embeddings[
            negative_image_indices,
            negative_caption_indices
        ]

        # Euclidean distances
        positive_distances = F.pairwise_distance(
            image_embeddings,
            positive_embeddings,
            p=2
        )

        negative_distances = F.pairwise_distance(
            image_embeddings,
            negative_embeddings,
            p=2
        )

        # Triplet margin loss
        loss = F.relu(
            positive_distances
            - negative_distances
            + self.margin
        ).mean()

        return loss


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Triplet Loss Tests")
    print("-" * 45)

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

    criterion = TripletContrastiveLoss(
        margin=0.2
    )

    loss = criterion(
        image_embeddings,
        text_embeddings
    )

    print(f"Device: {device}")
    print(f"Images: {image_embeddings.shape}")
    print(f"Captions: {text_embeddings.shape}")
    print(f"Loss: {loss.item():.4f}")

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0

    loss.backward()

    assert image_embeddings.grad is not None
    assert text_embeddings.grad is not None

    assert torch.isfinite(image_embeddings.grad).all()
    assert torch.isfinite(text_embeddings.grad).all()

    print("\nGradient checks passed!")

    # Test that a negative is never selected
    # from the same image as its anchor.
    for batch_size_test in [2, 4, 8]:

        indices = torch.arange(
            batch_size_test,
            device=device
        )

        offsets = torch.randint(
            1,
            batch_size_test,
            (batch_size_test,),
            device=device
        )

        negative_indices = (
            indices + offsets
        ) % batch_size_test

        assert torch.all(
            indices != negative_indices
        )

    print("Negative sampling checks passed!")

    print("\nAll Triplet Loss tests passed!")
