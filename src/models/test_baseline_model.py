
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.baseline_model import (
    BaselineImageEncoder,
    BaselineTextEncoder,
    BaselineRetrievalModel
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


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)

    print("Baseline Model Tests")
    print("-" * 45)

    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")

    batch_size = 2
    num_captions = 5
    sequence_length = 34

    images = torch.randn(
        batch_size, 3, 224, 224,
        device=device
    )

    captions = torch.randint(
        low=4,
        high=vocab_size,
        size=(batch_size, num_captions, sequence_length),
        device=device
    )

    # Add different amounts of padding.
    lengths = [25, 30, 20, 34, 28,
               22, 27, 18, 32, 24]

    for index, length in enumerate(lengths):
        image_index = index // num_captions
        caption_index = index % num_captions

        captions[
            image_index,
            caption_index,
            length:
        ] = 0

    model = BaselineRetrievalModel(
        vocab_size=vocab_size
    ).to(device)

    model.train()

    outputs = model(images, captions)

    image_embeddings = outputs["image_embeddings"]
    text_embeddings = outputs["text_embeddings"]

    # 1. Output shapes
    assert image_embeddings.shape == (2, 512)
    assert text_embeddings.shape == (2, 5, 512)

    print("Output shapes: PASS")

    # 2. Numerical validation
    assert torch.isfinite(image_embeddings).all()
    assert torch.isfinite(text_embeddings).all()

    print("Numerical validation: PASS")

    # 3. Embedding normalization
    image_norms = image_embeddings.norm(
        p=2, dim=1
    )

    text_norms = text_embeddings.norm(
        p=2, dim=2
    )

    assert torch.allclose(
        image_norms,
        torch.ones_like(image_norms),
        atol=1e-5
    )

    assert torch.allclose(
        text_norms,
        torch.ones_like(text_norms),
        atol=1e-5
    )

    print("Embedding normalization: PASS")

    # 4. Backward pass
    loss = 1 - (
        image_embeddings.unsqueeze(1) * text_embeddings
    ).sum(dim=2).mean()

    loss.backward()

    parameters_with_gradients = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_trainable_parameters = sum(
        parameter.requires_grad
        for parameter in model.parameters()
    )

    assert parameters_with_gradients == total_trainable_parameters

    print("Backward pass: PASS")

    # 5. Padding invariance
    text_encoder = BaselineTextEncoder(
        vocab_size=vocab_size
    ).to(device)

    text_encoder.eval()

    short_caption = torch.tensor(
        [[4, 5, 6, 7, 0, 0]],
        dtype=torch.long,
        device=device
    )

    longer_padding = torch.tensor(
        [[4, 5, 6, 7, 0, 0, 0, 0, 0]],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        short_embedding = text_encoder(short_caption)

        long_embedding = text_encoder(longer_padding)

    assert torch.allclose(
        short_embedding,
        long_embedding,
        atol=1e-6
    )

    print("Padding invariance: PASS")

    print("\nAll Baseline Model tests passed!")


if __name__ == "__main__":
    main()
