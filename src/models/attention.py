
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim, attention_dim=None):

        super().__init__()

        if attention_dim is None:
            attention_dim = hidden_dim

        self.feature_layer = nn.Linear(
            hidden_dim,
            attention_dim,
            bias=True
        )

        self.score_layer = nn.Linear(
            attention_dim,
            1,
            bias=False
        )

    def forward(self, hidden_states, mask=None):

        # hidden_states: [batch, sequence_length, hidden_dim]

        scores = self.score_layer(
            torch.tanh(
                self.feature_layer(hidden_states)
            )
        ).squeeze(-1)

        # scores: [batch, sequence_length]

        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min
            )

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        # Weighted sum of hidden states
        context_vector = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1),
            dim=1
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
        hidden_dim=hidden_dim
    ).to(device)

    # Simulate hidden states from an RNN
    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_dim
    ).to(device)

    # Simulate padding masks
    lengths = torch.tensor([34, 28, 25, 20])

    positions = torch.arange(
        sequence_length,
        device=device
    ).unsqueeze(0)

    mask = positions < lengths.to(device).unsqueeze(1)

    context, weights = model(
        hidden_states,
        mask
    )

    print("Additive Attention Test")
    print("-" * 40)

    print(f"Device: {device}")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Attention weights shape: {weights.shape}")

    print("\nAttention weight sums:")
    print(weights.sum(dim=1))

    print("\nPadding weights for sample 2:")
    print(weights[1, 28:])

    assert context.shape == (batch_size, hidden_dim)

    assert weights.shape == (
        batch_size,
        sequence_length
    )

    assert torch.allclose(
        weights.sum(dim=1),
        torch.ones(batch_size, device=device),
        atol=1e-6
    )

    assert torch.all(weights[1, 28:] == 0)

    assert torch.all(weights[2, 25:] == 0)

    assert torch.all(weights[3, 20:] == 0)

    print("\nAttention test passed!")
