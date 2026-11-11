
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPositiveInfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):

        super().__init__()

        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):

        # image_embeddings: [B, D]
        # text_embeddings: [B, C, D]

        batch_size, num_captions, embedding_dim = (
            text_embeddings.shape
        )

        if image_embeddings.shape != (
            batch_size,
            embedding_dim
        ):
            raise ValueError(
                "Image and text embedding dimensions must match"
            )

        if batch_size == 0 or num_captions == 0:
            raise ValueError("Batch and caption counts must be positive")

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

        # Flatten captions: [B, C, D] -> [B*C, D]
        flat_text_embeddings = text_embeddings.reshape(
            batch_size * num_captions,
            embedding_dim
        )

        # Similarity matrix: [B, B*C]
        logits = (
            image_embeddings @ flat_text_embeddings.T
        ) / self.temperature

        # Each image has C positive captions
        positive_mask = torch.zeros_like(
            logits,
            dtype=torch.bool
        )

        for image_idx in range(batch_size):

            start = image_idx * num_captions
            end = start + num_captions

            positive_mask[image_idx, start:end] = True

        # Image-to-Text loss
        image_to_text_log_probs = F.log_softmax(
            logits,
            dim=1
        )

        image_to_text_loss = -(
            image_to_text_log_probs.masked_fill(
                ~positive_mask,
                0.0
            ).sum(dim=1)
            / num_captions
        ).mean()

        # Text-to-Image similarity: [B*C, B]
        reverse_logits = logits.T

        # Every caption has exactly one positive image
        text_to_image_targets = torch.arange(
            batch_size,
            device=image_embeddings.device
        ).repeat_interleave(num_captions)

        text_to_image_loss = F.cross_entropy(
            reverse_logits,
            text_to_image_targets
        )

        # Symmetric contrastive loss
        loss = (
            image_to_text_loss + text_to_image_loss
        ) / 2

        return loss


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Multi-Positive InfoNCE Loss Tests")
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

    criterion = MultiPositiveInfoNCELoss(
        temperature=0.07
    )

    loss = criterion(
        image_embeddings,
        text_embeddings
    )

    print(f"Device: {device}")
    print(f"Image embeddings: {image_embeddings.shape}")
    print(f"Text embeddings: {text_embeddings.shape}")
    print(f"Loss: {loss.item():.4f}")

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert image_embeddings.grad is not None
    assert text_embeddings.grad is not None

    assert torch.isfinite(image_embeddings.grad).all()
    assert torch.isfinite(text_embeddings.grad).all()

    print("\nGradient checks passed!")

    print("\nAll Multi-Positive InfoNCE tests passed!")
