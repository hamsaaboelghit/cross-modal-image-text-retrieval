
import json
from pathlib import Path

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.full_model import CrossModalRetrievalModel

# Project paths
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

    print("Full Model Integration Test")
    print("-" * 50)

    # 1. Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # 2. Load vocabulary
    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 3. Model
    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=False,
        unfreeze_last_block=False,
    ).to(device)

    model.train()

    # 4. Dummy input
    batch_size = 2
    num_captions = 5
    max_length = 34

    images = torch.randn(
        batch_size, 3, 224, 224
    ).to(device)

    captions = torch.zeros(
        batch_size,
        num_captions,
        max_length,
        dtype=torch.long,
    )

    # Create captions with different lengths
    for i in range(batch_size):
        for j in range(num_captions):

            length = torch.randint(
                20, max_length + 1, (1,)
            ).item()

            tokens = torch.randint(
                4, vocab_size, (length,)
            )

            captions[i, j, :length] = tokens

    captions = captions.to(device)

    # 5. Forward pass
    outputs = model(
        images,
        captions,
        return_attention=True,
    )

    image_embeddings = outputs["image_embeddings"]
    text_embeddings = outputs["text_embeddings"]
    attention_weights = outputs["attention_weights"]

    print("\nForward Pass:")
    print(f"Image Embeddings: {image_embeddings.shape}")
    print(f"Text Embeddings: {text_embeddings.shape}")
    print(f"Attention Weights: {attention_weights.shape}")

    # 6. Shape validation
    assert image_embeddings.shape == (batch_size, 512)
    assert text_embeddings.shape == (
        batch_size, num_captions, 512
    )
    assert attention_weights.shape == (
        batch_size, num_captions, max_length
    )

    # 7. Embedding normalization
    image_norms = image_embeddings.norm(dim=1)
    text_norms = text_embeddings.norm(dim=2)

    assert torch.allclose(
        image_norms,
        torch.ones_like(image_norms),
        atol=1e-4,
    )

    assert torch.allclose(
        text_norms,
        torch.ones_like(text_norms),
        atol=1e-4,
    )

    print("\nEmbedding normalization: PASS")

    # 8. Numerical validation
    assert torch.isfinite(image_embeddings).all()
    assert torch.isfinite(text_embeddings).all()
    assert torch.isfinite(attention_weights).all()

    print("Numerical validation: PASS")

    # 9. Backward pass
    similarities = (
        image_embeddings.unsqueeze(1) * text_embeddings
    ).sum(dim=2)

    loss = 1.0 - similarities.mean()

    loss.backward()

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    params_with_grad = [
        p for p in trainable_params
        if p.grad is not None
    ]

    assert len(params_with_grad) > 0

    assert all(
        torch.isfinite(p.grad).all()
        for p in params_with_grad
    )

    print("\nBackward Pass: PASS")
    print(f"Test Loss: {loss.item():.6f}")
    print(
        f"Parameters with gradients: "
        f"{len(params_with_grad)}/{len(trainable_params)}"
    )

    print("\nAll Full Model integration tests passed!")


if __name__ == "__main__":
    main()
