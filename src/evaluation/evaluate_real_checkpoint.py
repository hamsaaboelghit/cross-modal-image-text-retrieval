
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.full_model import CrossModalRetrievalModel
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


def main():
    print("=" * 60)
    print("VISIONMATCH AI - BEST CHECKPOINT EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load vocabulary
    vocab_path = Path(
        "data/processed/flickr30k/vocabularies/fold_1_vocab.json"
    )

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {vocab_path}"
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 2. Load validation dataset (Fold 1)
    val_dataset = Flickr30kDataset(
        fold_number=1,
        split="validation",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Validation images: {len(val_dataset)}")

    # 3. Recreate the trained model architecture
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

    # 4. Load best checkpoint
    checkpoint_path = Path(
        "checkpoints/real_fold1/best_model.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nCheckpoint loaded successfully.")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Stored training loss: {checkpoint['train_loss']:.6f}")
    print(f"Stored validation loss: {checkpoint['val_loss']:.6f}")

    # 5. Evaluate retrieval performance
    evaluator = RetrievalEvaluator(
        model=model,
        device=device,
        query_batch_size=32,
        candidate_chunk_size=8192,
        ks=(1, 5, 10),
    )

    print("\nStarting validation retrieval evaluation...")

    results = evaluator.evaluate(val_loader)

    # 6. Display metrics
    print("\n" + "=" * 60)
    print("BEST CHECKPOINT RETRIEVAL RESULTS")
    print("=" * 60)

    for direction, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"\n{direction.upper().replace('_', ' ')}")

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.6f}")

    # 7. Save results
    output_dir = Path("results/real_fold1")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "best_model_retrieval_metrics.json"

    clean_results = {
        key: value
        for key, value in results.items()
        if key not in ("image_names", "names")
    }

    clean_results["checkpoint_epoch"] = checkpoint["epoch"]
    clean_results["training_loss"] = checkpoint["train_loss"]
    clean_results["validation_loss"] = checkpoint["val_loss"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4)

    print(f"\nMetrics saved to: {output_path}")
    print("\nValidation evaluation completed.")
    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
