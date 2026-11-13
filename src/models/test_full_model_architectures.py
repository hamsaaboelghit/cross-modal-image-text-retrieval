
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.full_model import CrossModalRetrievalModel


VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / "fold_1_vocab.json"
)


CONFIGURATIONS = [
    ("RNN", "rnn", False),
    ("LSTM", "lstm", False),
    ("GRU", "gru", False),
    ("Bidirectional RNN", "rnn", True),
    ("Bidirectional LSTM", "lstm", True),
    ("Bidirectional GRU", "gru", True),
    ("Transformer", "transformer", False),
]


def test_architecture(
    name,
    encoder_type,
    bidirectional,
    vocab_size,
    images,
    captions,
    device,
):
    print(f"\nTesting: {name}")
    print("-" * 45)

    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type=encoder_type,
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=bidirectional,
        use_attention=True,
        pretrained=False,
        unfreeze_last_block=False,
    ).to(device)

    model.train()

    # Forward pass
    outputs = model(
        images,
        captions,
        return_attention=True,
    )

    image_embeddings = outputs["image_embeddings"]
    text_embeddings = outputs["text_embeddings"]
    attention_weights = outputs["attention_weights"]

    # Expected shapes
    assert image_embeddings.shape == (2, 512)
    assert text_embeddings.shape == (2, 5, 512)
    assert attention_weights.shape == (2, 5, 34)

    # Embedding normalization
    assert torch.allclose(
        image_embeddings.norm(dim=1),
        torch.ones(2, device=device),
        atol=1e-4,
    )

    assert torch.allclose(
        text_embeddings.norm(dim=2),
        torch.ones(2, 5, device=device),
        atol=1e-4,
    )

    # Numerical validation
    assert torch.isfinite(image_embeddings).all()
    assert torch.isfinite(text_embeddings).all()
    assert torch.isfinite(attention_weights).all()

    # Attention must ignore padding tokens
    flat_captions = captions.reshape(
        -1,
        captions.size(2)
    )

    padding_mask = flat_captions == 0

    flat_attention = attention_weights.reshape(
        -1,
        captions.size(2)
    )

    assert torch.all(
        flat_attention[padding_mask] == 0
    ), "Attention assigned non-zero weights to padding tokens"

    # Backward pass
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

    print(f"Image embeddings: {image_embeddings.shape}")
    print(f"Text embeddings: {text_embeddings.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    print("Padding attention check: PASS")
    print(f"Loss: {loss.item():.6f}")
    print(
        f"Gradient check: "
        f"{len(params_with_grad)}/{len(trainable_params)}"
    )
    print("Architecture test: PASS")

    del model, outputs, loss

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():

    print("=" * 55)
    print("FULL MODEL - ARCHITECTURE INTEGRATION TESTS")
    print("=" * 55)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Shared dummy inputs
    batch_size = 2
    num_captions = 5
    max_length = 34

    images = torch.randn(
        batch_size, 3, 224, 224,
        device=device,
    )

    captions = torch.zeros(
        batch_size,
        num_captions,
        max_length,
        dtype=torch.long,
    )

    for i in range(batch_size):
        for j in range(num_captions):

            length = torch.randint(
                20, max_length + 1, (1,)
            ).item()

            captions[i, j, :length] = torch.randint(
                4, vocab_size, (length,)
            )

    captions = captions.to(device)

    passed = 0

    for name, encoder_type, bidirectional in CONFIGURATIONS:

        test_architecture(
            name=name,
            encoder_type=encoder_type,
            bidirectional=bidirectional,
            vocab_size=vocab_size,
            images=images,
            captions=captions,
            device=device,
        )

        passed += 1
            # Test model without Attention
    print("\nTesting: Model without Attention")
    print("-" * 45)

    no_attention_model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=False,
        pretrained=False,
        unfreeze_last_block=False,
    ).to(device)

    no_attention_model.eval()

    with torch.no_grad():
        no_attention_outputs = no_attention_model(
            images,
            captions,
            return_attention=True,
        )

    assert no_attention_outputs["image_embeddings"].shape == (2, 512)
    assert no_attention_outputs["text_embeddings"].shape == (2, 5, 512)
    assert no_attention_outputs["attention_weights"] is None

    print("Image embeddings: PASS")
    print("Text embeddings: PASS")
    print("Attention weights: None (as expected)")
    print("Model without Attention: PASS")

    del no_attention_model, no_attention_outputs

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 55)
    print(
        f"RESULT: {passed}/{len(CONFIGURATIONS)} "
        "ARCHITECTURES PASSED"
    )
    print("All architecture integration tests completed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
