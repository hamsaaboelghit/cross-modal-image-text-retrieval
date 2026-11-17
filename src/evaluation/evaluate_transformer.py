import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.full_model import CrossModalRetrievalModel
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


def main():
    print("=" * 60)
    print("VISIONMATCH AI - Transformer RETRIEVAL EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_path = Path(
        "data/processed/flickr30k/vocabularies/fold_1_vocab.json"
    )

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    val_dataset = Flickr30kDataset(1, "validation")

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CrossModalRetrievalModel(
        vocab_size=len(vocab),
        encoder_type="transformer",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=False,
        unfreeze_last_block=False,
    ).to(device)

    checkpoint_path = Path(
        "checkpoints/experiments/transformer_fold1_seed42/best_model.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded Transformer checkpoint: Epoch {checkpoint['epoch']}")
    print(f"Validation images: {len(val_dataset)}")

    evaluator = RetrievalEvaluator(
        model=model,
        device=device,
        query_batch_size=32,
        candidate_chunk_size=8192,
        ks=(1, 5, 10),
    )

    print("\nEvaluating Transformer on validation split...")
    results = evaluator.evaluate(val_loader)

    for direction, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"\n{direction.upper().replace('_', ' ')}")
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {name}: {value:.6f}")

    output_dir = Path("results/experiments/transformer_fold1_seed42")
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_results = {
        key: value
        for key, value in results.items()
        if key not in ("image_names", "names")
    }

    clean_results.update({
        "model_type": "ResNet50-Transformer",
        "checkpoint_epoch": checkpoint["epoch"],
        "training_loss": checkpoint["train_loss"],
        "validation_loss": checkpoint["val_loss"],
        "seed": 42,
        "split": "validation",
    })

    output_path = output_dir / "retrieval_metrics.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4)

    print(f"\nResults saved to: {output_path}")
    print("Validation evaluation completed. Test split was NOT used.")


if __name__ == "__main__":
    main()

