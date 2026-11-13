
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

try:
    from .attention import AdditiveAttention
except ImportError:
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


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        hidden_dim,
        dropout=0.1,
        max_length=512,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError(
                "hidden_dim must be even for sinusoidal positional encoding"
            )

        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2)
            * (-math.log(10000.0) / hidden_dim)
        )

        positional_encoding = torch.zeros(
            max_length,
            hidden_dim,
        )

        positional_encoding[:, 0::2] = torch.sin(
            positions * div_term
        )

        positional_encoding[:, 1::2] = torch.cos(
            positions * div_term
        )

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer(
            "positional_encoding",
            positional_encoding,
        )

    def forward(self, embeddings):

        sequence_length = embeddings.size(1)

        if sequence_length > self.positional_encoding.size(1):
            raise ValueError(
                "Sequence length exceeds maximum positional encoding length"
            )

        embeddings = (
            embeddings
            + self.positional_encoding[:, :sequence_length, :]
        )

        return self.dropout(embeddings)


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
        use_attention=False,
    ):

        super().__init__()

        encoder_type = encoder_type.lower()

        if encoder_type not in {
            "rnn",
            "lstm",
            "gru",
            "transformer",
        }:
            raise ValueError(
                "encoder_type must be rnn, lstm, gru, or transformer"
            )

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.num_directions = (
            2 if bidirectional else 1
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_IDX,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # Classic architectures: RNN, LSTM, GRU
        if encoder_type in {"rnn", "lstm", "gru"}:

            rnn_args = {
                "input_size": embedding_dim,
                "hidden_size": hidden_dim,
                "num_layers": num_layers,
                "batch_first": True,
                "bidirectional": bidirectional,
                "dropout": dropout if num_layers > 1 else 0.0,
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

            self.transformer_encoder = None
            self.input_projection = None
            self.positional_encoding = None

        # Modern architecture: Transformer
        else:

            if bidirectional:
                raise ValueError(
                    "bidirectional is not applicable to Transformer"
                )

            if hidden_dim % 8 != 0:
                raise ValueError(
                    "hidden_dim must be divisible by 8 for Transformer"
                )

            self.rnn = None

            self.input_projection = nn.Linear(
                embedding_dim,
                hidden_dim,
            )

            self.positional_encoding = PositionalEncoding(
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

            transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )

            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=transformer_layer,
                num_layers=num_layers,
            )

            self.encoder_output_dim = hidden_dim

        # Shared output dimension
        if use_attention:

            self.attention = AdditiveAttention(
                hidden_dim=self.encoder_output_dim
            )

        else:
            self.attention = None

        self.output_dropout = nn.Dropout(dropout)

        self.output_dim = self.encoder_output_dim

    def forward(self, captions, return_attention=False):

        if captions.ndim != 2:
            raise ValueError(
                "Captions must have shape [batch, sequence_length]"
            )

        lengths = (
            captions != PAD_IDX
        ).sum(dim=1).cpu()

        if torch.any(lengths == 0):
            raise ValueError(
                "Every caption must contain at least one token"
            )

        # Padding mask: True means padding
        padding_mask = captions == PAD_IDX

        # Embedding
        embedded = self.embedding(captions)
        embedded = self.embedding_dropout(embedded)

        # Encoder
        if self.encoder_type == "transformer":

            projected = self.input_projection(embedded)

            positioned = self.positional_encoding(projected)

            outputs = self.transformer_encoder(
                positioned,
                src_key_padding_mask=padding_mask,
            )

            hidden = None

        else:

            packed = pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )

            packed_outputs, hidden = self.rnn(packed)

            outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=captions.size(1),
            )

            if self.encoder_type == "lstm":
                hidden = hidden[0]

        # Build valid-token mask
        valid_mask = ~padding_mask

        # Attention or pooling
        if self.use_attention:

            features, attention_weights = self.attention(
                outputs,
                valid_mask,
            )

        else:

            attention_weights = None

            if self.encoder_type == "transformer":

                valid_mask_float = valid_mask.unsqueeze(-1).to(
                    outputs.dtype
                )

                features = (
                    (outputs * valid_mask_float).sum(dim=1)
                    / valid_mask_float.sum(dim=1).clamp(min=1)
                )

            elif self.bidirectional:

                features = torch.cat(
                    [hidden[-2], hidden[-1]],
                    dim=1,
                )

            else:

                features = hidden[-1]

        features = self.output_dropout(features)

        if return_attention:
            return features, attention_weights

        return features


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)

    dummy_captions = torch.randint(
        low=4,
        high=vocab_size,
        size=(5, 34),
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
        ("transformer", False, False),
        ("transformer", False, True),
    ]

    print("Text Encoder Architecture Tests")
    print("-" * 55)
    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")

    for encoder_type, bidirectional, use_attention in configurations:

        model = TextEncoder(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            embedding_dim=300,
            hidden_dim=512,
            num_layers=1,
            dropout=0.2,
            bidirectional=bidirectional,
            use_attention=use_attention,
        ).to(device)

        model.eval()

        with torch.no_grad():

            features, weights = model(
                dummy_captions,
                return_attention=True,
            )

        expected_dim = (
            512 * (2 if bidirectional else 1)
        )

        assert features.shape == (5, expected_dim)

        assert torch.isfinite(features).all()

        if use_attention:

            assert weights.shape == (5, 34)

            assert torch.allclose(
                weights.sum(dim=1),
                torch.ones(5, device=device),
                atol=1e-6,
            )

            assert torch.all(
                weights[dummy_captions == PAD_IDX] == 0
            )

        else:
            assert weights is None

        print(
            f"{encoder_type.upper():11} | "
            f"Bi={bidirectional} | "
            f"Attention={use_attention} | "
            f"Output={list(features.shape)} | PASS"
        )

        del model, features, weights

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nAll Text Encoder tests passed!")
