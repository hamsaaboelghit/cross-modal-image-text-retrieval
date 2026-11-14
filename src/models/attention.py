
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim, attention_dim=None):

        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if attention_dim is None:
            attention_dim = hidden_dim

        if attention_dim <= 0:
            raise ValueError(
                "attention_dim must be greater than zero."
            )

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.feature_layer = nn.Linear(
            hidden_dim,
            attention_dim,
            bias=True,
        )

        self.score_layer = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )

    def forward(self, hidden_states, mask=None):

        # Input Validation
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(
                "hidden_states must be a PyTorch tensor."
            )

        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch, sequence_length, hidden_dim]."
            )

        batch_size, sequence_length, hidden_dim = (
            hidden_states.shape
        )

        if batch_size == 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if sequence_length == 0:
            raise ValueError(
                "Sequence length must be greater than zero."
            )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"but received {hidden_dim}."
            )

        if not torch.is_floating_point(hidden_states):
            raise TypeError(
                "hidden_states must have a floating-point dtype."
            )

        if not torch.isfinite(hidden_states).all():
            raise ValueError(
                "hidden_states contains NaN or Inf values."
            )

        # Mask Validation
        if mask is not None:

            if not isinstance(mask, torch.Tensor):
                raise TypeError(
                    "mask must be a PyTorch tensor."
                )

            if mask.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "mask must have shape "
                    "[batch, sequence_length]."
                )

            if mask.dtype != torch.bool:
                raise TypeError(
                    "mask must have boolean dtype."
                )

            if mask.device != hidden_states.device:
                raise ValueError(
                    "mask and hidden_states must be "
                    "on the same device."
                )

            # Every sample must contain at least one valid token.
            valid_samples = mask.any(dim=1)

            if not torch.all(valid_samples):
                raise ValueError(
                    "Every sample must contain at least "
                    "one valid token. All-padding sequences "
                    "are not supported."
                )

        # 1. Calculate Additive Attention Scores
        scores = self.score_layer(
            torch.tanh(
                self.feature_layer(hidden_states)
            )
        ).squeeze(-1)

        # scores: [batch, sequence_length]

        # 2. Apply Padding Mask
        if mask is not None:

            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min,
            )

        # 3. Normalize Attention Scores
        attention_weights = torch.softmax(
            scores,
            dim=1,
        )

        # 4. Calculate Context Vector
        context_vector = torch.sum(
            hidden_states
            * attention_weights.unsqueeze(-1),
            dim=1,
        )

        return context_vector, attention_weights


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 4
    sequence_length = 34
    hidden_dim = 512

    model = AdditiveAttention(
        hidden_dim=hidden_dim,
    ).to(device)

    # --------------------------------------------------
    # Test 1: Normal Forward Pass
    # --------------------------------------------------

    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_dim,
        device=device,
    )

    lengths = torch.tensor(
        [34, 28, 25, 20],
        device=device,
    )

    positions = torch.arange(
        sequence_length,
        device=device,
    ).unsqueeze(0)

    mask = positions < lengths.unsqueeze(1)

    context, weights = model(
        hidden_states,
        mask,
    )

    assert context.shape == (
        batch_size,
        hidden_dim,
    )

    assert weights.shape == (
        batch_size,
        sequence_length,
    )

    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()

    print("Test 1 - Forward Pass: PASS")

    # --------------------------------------------------
    # Test 2: Attention Normalization
    # --------------------------------------------------

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(batch_size, device=device),
        atol=1e-6,
    )

    print("Test 2 - Attention Normalization: PASS")

    # --------------------------------------------------
    # Test 3: Padding Mask
    # --------------------------------------------------

    for i, length in enumerate(lengths.tolist()):

        assert torch.all(
            weights[i, length:] == 0
        )

    print("Test 3 - Padding Mask: PASS")

    # --------------------------------------------------
    # Test 4: Backward Pass
    # --------------------------------------------------

    model.zero_grad()

    loss = context.square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():

        assert parameter.grad is not None, (
            f"Missing gradient: {name}"
        )

        assert torch.isfinite(parameter.grad).all(), (
            f"Invalid gradient: {name}"
        )

    print("Test 4 - Backward Pass: PASS")

    # --------------------------------------------------
    # Test 5: All-Padding Input Rejection
    # --------------------------------------------------

    all_padding_mask = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.bool,
        device=device,
    )

    try:

        model(
            hidden_states,
            all_padding_mask,
        )

        raise AssertionError(
            "All-padding input should have been rejected."
        )

    except ValueError as error:

        assert "All-padding sequences" in str(error)

    print("Test 5 - All-Padding Rejection: PASS")

    # --------------------------------------------------
    # Test 6: Invalid Dimensions
    # --------------------------------------------------

    try:

        model(
            torch.randn(4, 512, device=device),
        )

        raise AssertionError(
            "Invalid input dimensions should be rejected."
        )

    except TypeError:
        raise

    except ValueError:
        pass

    print("Test 6 - Invalid Input Validation: PASS")

    # --------------------------------------------------
    # Final Results
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("ADDITIVE ATTENTION TEST RESULTS")
    print("=" * 50)

    print(f"Device: {device}")
    print(f"Input Shape: {hidden_states.shape}")
    print(f"Context Shape: {context.shape}")
    print(f"Attention Shape: {weights.shape}")
    print("All tests passed!")
