
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Text-to-Image Multi-Head Cross-Attention.

    Text tokens act as Queries.
    Image regions act as Keys and Values.

    Inputs:
        text_features: [B, L, text_dim]
        image_regions: [B, R, image_dim]
        text_mask:     [B, L], True for valid tokens

    Outputs:
        output_features: [B, L, embedding_dim]
        attention_weights: [B, L, R]
    """

    def __init__(
        self,
        text_dim,
        image_dim=2048,
        embedding_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Project text tokens into the shared embedding space.
        self.query_projection = nn.Linear(
            text_dim,
            embedding_dim,
        )

        # Project image regions into the shared embedding space.
        self.key_projection = nn.Linear(
            image_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            image_dim,
            embedding_dim,
        )

        # Output projection.
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Residual connection for text features.
        self.residual_projection = nn.Linear(
            text_dim,
            embedding_dim,
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(
        self,
        text_features,
        image_regions,
        text_mask=None,
    ):

        if text_features.ndim != 3:
            raise ValueError(
                "text_features must have shape [B, L, text_dim]."
            )

        if image_regions.ndim != 3:
            raise ValueError(
                "image_regions must have shape [B, R, image_dim]."
            )

        batch_size, sequence_length, text_dim = (
            text_features.shape
        )

        image_batch_size, num_regions, image_dim = (
            image_regions.shape
        )

        if text_dim != self.text_dim:
            raise ValueError(
                f"Expected text_dim={self.text_dim}, got {text_dim}."
            )

        if image_dim != self.image_dim:
            raise ValueError(
                f"Expected image_dim={self.image_dim}, got {image_dim}."
            )

        if batch_size != image_batch_size:
            raise ValueError(
                "Text and image batches must have the same size."
            )

        if num_regions == 0:
            raise ValueError(
                "At least one image region is required."
            )

        if text_mask is not None:

            if text_mask.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "text_mask must have shape [B, L]."
                )

            if text_mask.dtype != torch.bool:
                raise TypeError(
                    "text_mask must have boolean dtype."
                )

            if not torch.all(text_mask.any(dim=1)):
                raise ValueError(
                    "Every sample must contain at least one valid token."
                )

        # 1. Generate Queries, Keys and Values.
        queries = self.query_projection(
            text_features
        )

        keys = self.key_projection(
            image_regions
        )

        values = self.value_projection(
            image_regions
        )

        # 2. Split into multiple attention heads.
        # Q: [B, heads, L, head_dim]
        # K: [B, heads, R, head_dim]
        # V: [B, heads, R, head_dim]

        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        keys = keys.reshape(
            batch_size,
            num_regions,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        values = values.reshape(
            batch_size,
            num_regions,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # 3. Scaled Dot-Product Attention.
        attention_scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )

        attention_scores = attention_scores / (
            self.head_dim ** 0.5
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        # Preserve normalized attention weights for interpretation.
        normalized_attention_weights = attention_weights

        # Apply dropout only for context computation.
        dropped_attention_weights = self.attention_dropout(
            attention_weights
        )

        # 4. Combine image information for each text token.
        context = torch.matmul(
            dropped_attention_weights,
            values,
        )

        # [B, heads, L, head_dim]
        # -> [B, L, embedding_dim]
        context = context.transpose(
            1, 2
        ).contiguous().reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # 5. Output projection.
        context = self.output_projection(
            context
        )

        # 6. Residual connection.
        residual = self.residual_projection(
            text_features
        )

        output_features = self.layer_norm(
            residual + self.output_dropout(context)
        )

        # 7. Ignore padded text positions.
        if text_mask is not None:

            output_features = output_features.masked_fill(
                ~text_mask.unsqueeze(-1),
                0.0,
            )

            normalized_attention_weights = (
                normalized_attention_weights.masked_fill(
                    ~text_mask[:, None, :, None],
                    0.0,
                )
            )

        # Average normalized attention across heads.
        # [B, heads, L, R] -> [B, L, R]
        averaged_attention = normalized_attention_weights.mean(
            dim=1
        )

        return output_features, averaged_attention


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Cross-Modal Attention Tests")
    print("-" * 55)
    print(f"Device: {device}")

    batch_size = 4
    sequence_length = 34
    num_regions = 49
    image_dim = 2048
    embedding_dim = 512

    # Test both unidirectional and bidirectional text dimensions.
    text_dimensions = [512, 1024]

    for text_dim in text_dimensions:

        print(f"\nTesting text_dim={text_dim}")

        model = CrossModalAttention(
            text_dim=text_dim,
            image_dim=image_dim,
            embedding_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
        ).to(device)

        text_features = torch.randn(
            batch_size,
            sequence_length,
            text_dim,
            device=device,
        )

        image_regions = torch.randn(
            batch_size,
            num_regions,
            image_dim,
            device=device,
        )

        # Simulate different caption lengths.
        text_mask = torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=device,
        )

        lengths = [34, 28, 20, 15]

        for i, length in enumerate(lengths):
            text_mask[i, length:] = False

        # --------------------------------
        # Evaluation Mode Tests
        # --------------------------------

        model.eval()

        with torch.no_grad():

            output_features, attention_weights = model(
                text_features,
                image_regions,
                text_mask,
            )

        # Shape checks.
        assert output_features.shape == (
            batch_size,
            sequence_length,
            embedding_dim,
        )

        assert attention_weights.shape == (
            batch_size,
            sequence_length,
            num_regions,
        )

        # Numerical checks.
        assert torch.isfinite(output_features).all()
        assert torch.isfinite(attention_weights).all()

        # Attention should sum to 1 for valid tokens.
        valid_attention_sums = attention_weights.sum(
            dim=-1
        )

        assert torch.allclose(
            valid_attention_sums[text_mask],
            torch.ones_like(
                valid_attention_sums[text_mask]
            ),
            atol=1e-5,
        )

        # Padded token attention must be zero.
        assert torch.all(
            attention_weights[~text_mask] == 0
        )

        # Padded output features must be zero.
        assert torch.all(
            output_features[~text_mask] == 0
        )

        # --------------------------------
        # Training Mode Tests
        # --------------------------------

        model.train()

        output_features, attention_weights = model(
            text_features,
            image_regions,
            text_mask,
        )

        # Attention must remain normalized during training.
        train_attention_sums = attention_weights.sum(
            dim=-1
        )

        assert torch.allclose(
            train_attention_sums[text_mask],
            torch.ones_like(
                train_attention_sums[text_mask]
            ),
            atol=1e-5,
        )

        # Padded attention must remain zero during training.
        assert torch.all(
            attention_weights[~text_mask] == 0
        )

        # Backward pass.
        loss = output_features.square().mean()

        loss.backward()

        # Check gradients.
        parameters_with_gradients = sum(
            1
            for parameter in model.parameters()
            if parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
        )

        total_parameters = sum(
            1
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        assert parameters_with_gradients == total_parameters

        print(
            f"Output Features: {list(output_features.shape)} | PASS"
        )

        print(
            f"Attention Weights: {list(attention_weights.shape)} | PASS"
        )

        print("Evaluation Mode: PASS")
        print("Training Mode: PASS")
        print("Padding Mask: PASS")
        print("Attention Normalization: PASS")
        print("Backward Pass: PASS")

        print(
            f"Gradient Check: "
            f"{parameters_with_gradients}/{total_parameters} | PASS"
        )

        del model
        del text_features, image_regions
        del output_features, attention_weights

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nAll Cross-Modal Attention tests passed!")
