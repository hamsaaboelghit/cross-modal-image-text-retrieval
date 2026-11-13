
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_IDX = 0


class BaselineImageEncoder(nn.Module):

    def __init__(self, output_dim=256):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output_projection = nn.Linear(
            256,
            output_dim
        )

        self.output_dim = output_dim

    def forward(self, images):

        features = self.features(images)

        features = torch.flatten(
            features,
            start_dim=1
        )

        embeddings = self.output_projection(features)

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings


class BaselineTextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        output_dim=256,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX
        )

        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(
            embedding_dim,
            output_dim
        )

        self.output_dim = output_dim

    def forward(self, captions):

        if captions.ndim != 2:
            raise ValueError(
                "Captions must have shape [N, L]."
            )

        mask = captions != PAD_IDX

        lengths = mask.sum(dim=1, keepdim=True)

        if torch.any(lengths == 0):
            raise ValueError(
                "Every caption must contain at least one token."
            )

        embeddings = self.embedding(captions)

        embeddings = embeddings * mask.unsqueeze(-1)

        pooled = embeddings.sum(dim=1) / lengths

        pooled = self.dropout(pooled)

        projected = self.output_projection(pooled)

        projected = F.normalize(
            projected,
            p=2,
            dim=1
        )

        return projected


class BaselineRetrievalModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        output_dim=512,
        dropout=0.1
    ):

        super().__init__()

        self.image_encoder = BaselineImageEncoder(
            output_dim=output_dim
        )

        self.text_encoder = BaselineTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            dropout=dropout
        )

        self.embedding_dim = output_dim

    def forward(self, images, captions):

        if images.ndim != 4:
            raise ValueError(
                "Images must have shape [B, 3, H, W]."
            )

        if captions.ndim != 3:
            raise ValueError(
                "Captions must have shape [B, C, L]."
            )

        batch_size = images.size(0)
        num_captions = captions.size(1)

        if captions.size(0) != batch_size:
            raise ValueError(
                "Images and captions must have the same batch size."
            )

        image_embeddings = self.image_encoder(images)

        flat_captions = captions.reshape(
            batch_size * num_captions,
            captions.size(2)
        )

        flat_text_embeddings = self.text_encoder(
            flat_captions
        )

        text_embeddings = flat_text_embeddings.reshape(
            batch_size,
            num_captions,
            self.embedding_dim
        )

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings
        }
