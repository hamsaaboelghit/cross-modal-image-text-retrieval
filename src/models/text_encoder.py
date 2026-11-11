
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)

from attention import AdditiveAttention


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / "fold_1_vocab.json"
)

PAD_IDX = 0


class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=False
    ):

        super().__init__()

        encoder_type = encoder_type.lower()

        if encoder_type not in {"rnn", "lstm", "gru"}:
            raise ValueError(
                "encoder_type must be rnn, lstm, or gru"
            )

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX
        )

        self.embedding_dropout = nn.Dropout(dropout)

        rnn_args = {
            "input_size": embedding_dim,
            "hidden_size": hidden_dim,
            "num_layers": num_layers,
            "batch_first": True,
            "bidirectional": bidirectional,
            "dropout": dropout if num_layers > 1 else 0.0
        }

        if encoder_type == "rnn":
            self.rnn = nn.RNN(**rnn_args)

        elif encoder_type == "lstm":
            self.rnn = nn.LSTM(**rnn_args)

        else:
            self.rnn = nn.GRU(**rnn_args)

        self.encoder_output_dim = (
            hidden_dim * self.num_directions
        )

        if use_attention:
            self.attention = AdditiveAttention(
                hidden_dim=self.encoder_output_dim
            )

        self.output_dropout = nn.Dropout(dropout)

        self.output_dim = self.encoder_output_dim

    def forward(self, captions, return_attention=False):

        lengths = (captions != PAD_IDX).sum(dim=1).cpu()

        if torch.any(lengths == 0):
            raise ValueError(
                "Every caption must contain at least one token"
            )

        embedded = self.embedding(captions)
        embedded = self.embedding_dropout(embedded)

        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_outputs, hidden = self.rnn(packed)

        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=captions.size(1)
        )

        if self.encoder_type == "lstm":
            hidden = hidden[0]

        if self.use_attention:

            positions = torch.arange(
                captions.size(1),
                device=captions.device
            ).unsqueeze(0)

            mask = positions < lengths.to(
                captions.device
            ).unsqueeze(1)

            features, attention_weights = self.attention(
                outputs,
                mask
            )

        else:

            if self.bidirectional:
                features = torch.cat(
                    [hidden[-2], hidden[-1]],
                    dim=1
                )
            else:
                features = hidden[-1]

            attention_weights = None

        features = self.output_dropout(features)

        if return_attention:
            return features, attention_weights

        return features


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)

    dummy_captions = torch.randint(
        low=4,
        high=vocab_size,
        size=(5, 34)
    )

    lengths = [34, 25, 30, 20, 28]

    for i, length in enumerate(lengths):
        dummy_captions[i, length:] = PAD_IDX

    dummy_captions = dummy_captions.to(device)

    configurations = [
        ("rnn", False, False),
        ("lstm", False, False),
        ("gru", False, False),
        ("rnn", False, True),
        ("lstm", False, True),
        ("gru", False, True),
        ("rnn", True, True),
        ("lstm", True, True),
        ("gru", True, True),
    ]

    print("Text Encoder Architecture Tests")
    print("-" * 45)

    for encoder_type, bidirectional, use_attention in configurations:

        model = TextEncoder(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            use_attention=use_attention
        ).to(device)

        if use_attention:
            features, weights = model(
                dummy_captions,
                return_attention=True
            )
        else:
            features = model(dummy_captions)
            weights = None

        expected_dim = 512 * (2 if bidirectional else 1)

        assert features.shape == (5, expected_dim)

        if use_attention:
            assert weights.shape == (5, 34)

            assert torch.allclose(
                weights.sum(dim=1),
                torch.ones(5, device=device),
                atol=1e-6
            )

        print(
            f"{encoder_type.upper():4} | "
            f"Bi={bidirectional} | "
            f"Attention={use_attention} | "
            f"Output={list(features.shape)} | PASS"
        )

        del model, features, weights

    print("\nAll Text Encoder tests passed!")
