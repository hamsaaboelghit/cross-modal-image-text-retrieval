
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


def main():

    print("=" * 60)
    print("FULL MODEL - CROSS-MODAL MATCHING TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    vocab_size = len(vocab)

    batch_size = 2
    num_captions = 5
    sequence_length = 34

    # --------------------------------------------------
    # 1. Create Dummy Inputs
    # --------------------------------------------------

    images = torch.randn(
        batch_size,
        3,
        224,
        224,
        device=device,
    )

    captions = torch.zeros(
        batch_size,
        num_captions,
        sequence_length,
        dtype=torch.long,
        device=device,
    )

    for i in range(batch_size):
        for j in range(num_captions):

            length = torch.randint(
                15,
                sequence_length + 1,
                (1,),
            ).item()

            captions[i, j, :length] = torch.randint(
                4,
                vocab_size,
                (length,),
                device=device,
            )

    # --------------------------------------------------
    # 2. Initialize Full Model
    # --------------------------------------------------

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
        use_cross_modal_attention=True,
    ).to(device)

    # --------------------------------------------------
    # 3. Evaluation Forward Pass
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(
            images,
            captions,
            return_attention=True,
        )

    image_embeddings = outputs["image_embeddings"]
    text_embeddings = outputs["text_embeddings"]
    matching_scores = outputs["matching_scores"]

    text_attention = outputs["attention_weights"]

    cross_attention = outputs[
        "cross_modal_attention_weights"
    ]

    # --------------------------------------------------
    # 4. Shape Checks
    # --------------------------------------------------

    assert image_embeddings.shape == (2, 512)

    assert text_embeddings.shape == (2, 5, 512)

    assert matching_scores.shape == (2, 5)

    assert text_attention.shape == (2, 5, 34)

    assert cross_attention.shape == (2, 5, 34, 49)

    # --------------------------------------------------
    # 5. Numerical Checks
    # --------------------------------------------------

    assert torch.isfinite(image_embeddings).all()

    assert torch.isfinite(text_embeddings).all()

    assert torch.isfinite(matching_scores).all()

    assert torch.isfinite(text_attention).all()

    assert torch.isfinite(cross_attention).all()

    # --------------------------------------------------
    # 6. Embedding Normalization
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 7. Padding Checks
    # --------------------------------------------------

    flat_captions = captions.reshape(
        -1,
        sequence_length,
    )

    padding_mask = flat_captions == 0

    flat_text_attention = text_attention.reshape(
        -1,
        sequence_length,
    )

    flat_cross_attention = cross_attention.reshape(
        -1,
        sequence_length,
        49,
    )

    assert torch.all(
        flat_text_attention[padding_mask] == 0
    )

    assert torch.all(
        flat_cross_attention[padding_mask] == 0
    )

    # --------------------------------------------------
    # 8. Cross-Attention Normalization
    # --------------------------------------------------

    valid_mask = ~padding_mask

    attention_sums = flat_cross_attention.sum(
        dim=-1
    )

    assert torch.allclose(
        attention_sums[valid_mask],
        torch.ones_like(
            attention_sums[valid_mask]
        ),
        atol=1e-4,
    )

    # --------------------------------------------------
    # 9. Training Forward Pass
    # --------------------------------------------------

    model.train()

    train_outputs = model(
        images,
        captions,
        return_attention=False,
    )

    train_image_embeddings = train_outputs[
        "image_embeddings"
    ]

    train_text_embeddings = train_outputs[
        "text_embeddings"
    ]

    train_matching_scores = train_outputs[
        "matching_scores"
    ]

    # --------------------------------------------------
    # 10. Backward Pass
    # --------------------------------------------------

    retrieval_similarity = (
        train_image_embeddings.unsqueeze(1)
        * train_text_embeddings
    ).sum(dim=2)

    loss = (
        1.0 - train_matching_scores.mean()
        + 1.0 - retrieval_similarity.mean()
    )

    assert loss.requires_grad

    loss.backward()

    # --------------------------------------------------
    # 11. Gradient Checks
    # --------------------------------------------------

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    params_with_grad = [
        p for p in trainable_params
        if p.grad is not None
        and torch.isfinite(p.grad).all()
    ]

    assert len(params_with_grad) == len(trainable_params), (
        f"Gradient mismatch: "
        f"{len(params_with_grad)}/{len(trainable_params)}"
    )

    # --------------------------------------------------
    # 12. Results
    # --------------------------------------------------

    print("\nOutput Shapes:")

    print(
        f"Image Embeddings: "
        f"{list(image_embeddings.shape)}"
    )

    print(
        f"Text Embeddings: "
        f"{list(text_embeddings.shape)}"
    )

    print(
        f"Matching Scores: "
        f"{list(matching_scores.shape)}"
    )

    print(
        f"Text Attention: "
        f"{list(text_attention.shape)}"
    )

    print(
        f"Cross-Modal Attention: "
        f"{list(cross_attention.shape)}"
    )

    print("\nValidation:")

    print("Numerical checks: PASS")

    print("Embedding normalization: PASS")

    print("Padding checks: PASS")

    print("Cross-attention normalization: PASS")

    print("Backward pass: PASS")

    print(
        f"Gradient check: "
        f"{len(params_with_grad)}/"
        f"{len(trainable_params)} | PASS"
    )

    print(
        f"\nTest Loss: {loss.item():.6f}"
    )

    print("\nAll Cross-Modal Full Model tests passed!")

    del model
    del outputs
    del train_outputs
    del loss

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
