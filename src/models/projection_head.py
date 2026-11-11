
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim=512,
        hidden_dim=None,
        dropout=0.1
    ):

        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.projection = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim)

        )

        self.output_dim = embedding_dim

    def forward(self, features):

        embeddings = self.projection(features)

        # Normalize embeddings to unit length
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings


class CrossModalProjection(nn.Module):

    def __init__(
        self,
        image_dim=2048,
        text_dim=512,
        embedding_dim=512,
        dropout=0.1
    ):

        super().__init__()

        self.image_projection = ProjectionHead(
            input_dim=image_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.text_projection = ProjectionHead(
            input_dim=text_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.embedding_dim = embedding_dim

    def forward(self, image_features, text_features):

        image_embeddings = self.image_projection(
            image_features
        )

        text_embeddings = self.text_projection(
            text_features
        )

        return image_embeddings, text_embeddings


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 8
    embedding_dim = 512

    print("Cross-Modal Projection Tests")
    print("-" * 45)

    configurations = [
        (2048, 512),
        (2048, 1024),
        (2048, 512)
    ]

    for image_dim, text_dim in configurations:

        model = CrossModalProjection(
            image_dim=image_dim,
            text_dim=text_dim,
            embedding_dim=embedding_dim
        ).to(device)

        image_features = torch.randn(
            batch_size,
            image_dim,
            device=device
        )

        text_features = torch.randn(
            batch_size,
            text_dim,
            device=device
        )

        image_embeddings, text_embeddings = model(
            image_features,
            text_features
        )

        assert image_embeddings.shape == (
            batch_size,
            embedding_dim
        )

        assert text_embeddings.shape == (
            batch_size,
            embedding_dim
        )

        image_norms = torch.norm(
            image_embeddings,
            p=2,
            dim=1
        )

        text_norms = torch.norm(
            text_embeddings,
            p=2,
            dim=1
        )

        assert torch.allclose(
            image_norms,
            torch.ones(batch_size, device=device),
            atol=1e-5
        )

        assert torch.allclose(
            text_norms,
            torch.ones(batch_size, device=device),
            atol=1e-5
        )

        similarity = image_embeddings @ text_embeddings.T

        assert similarity.shape == (
            batch_size,
            batch_size
        )

        print(
            f"Image: {image_dim} -> {embedding_dim} | "
            f"Text: {text_dim} -> {embedding_dim} | PASS"
        )

        print(
            f"Similarity Matrix: {list(similarity.shape)}"
        )

        del model

    print("\nAll Projection Head tests passed!")
