
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence
)


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
        dropout=0.2
    ):

        super().__init__()

        encoder_type = encoder_type.lower()

        if encoder_type not in {"rnn", "lstm", "gru"}:
            raise ValueError(
                "encoder_type must be rnn, lstm, or gru"
            )

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )

        self.embedding_dropout = nn.Dropout(dropout)

        if encoder_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )

        elif encoder_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )

        else:
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )

        self.output_dropout = nn.Dropout(dropout)

        self.output_dim = hidden_dim

    def forward(self, captions):

        # captions shape: [batch_size, sequence_length]

        lengths = (
            captions != PAD_IDX
        ).sum(dim=1).cpu()

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

        _, hidden = self.rnn(packed)

        # LSTM returns (hidden_state, cell_state)
        if self.encoder_type == "lstm":
            hidden = hidden[0]

        # Last layer's final hidden state
        features = hidden[-1]

        features = self.output_dropout(features)

        return features


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)

    print("Text Encoder Test")
    print("-" * 40)
    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")

    # Test with 5 captions, each of length 34
    dummy_captions = torch.randint(
        low=4,
        high=vocab_size,
        size=(5, 34)
    )

    # Add padding to simulate real captions
    dummy_captions[0, 25:] = PAD_IDX
    dummy_captions[1, 30:] = PAD_IDX
    dummy_captions[2, 28:] = PAD_IDX

    dummy_captions = dummy_captions.to(device)

    for encoder_type in ["rnn", "lstm", "gru"]:

        model = TextEncoder(
            vocab_size=vocab_size,
            encoder_type=encoder_type
        ).to(device)

        features = model(dummy_captions)

        print(f"\n{encoder_type.upper()} Results")
        print(f"Input shape: {dummy_captions.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Output dimension: {model.output_dim}")

        assert features.shape == (5, 512)

        del model, features

    print("\nAll Text Encoder tests passed!")
