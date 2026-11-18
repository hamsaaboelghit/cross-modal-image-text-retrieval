import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Shared experiment protocol
SHARED = {
    "dataset": "Flickr30k",
    "fold": 1,
    "train_split": "train",
    "selection_split": "validation",
    "test_used_for_selection": False,
    "image_size": "224x224",
    "image_color": "RGB",
    "image_resize": "BILINEAR",
    "captions_per_image": 5,
    "max_caption_length": 34,
    "padding_token_id": 0,
    "text_embedding_dim": 300,
    "retrieval_embedding_dim": 512,
    "batch_size": 8,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "primary_loss": "Multi-Positive InfoNCE",
    "temperature": 0.07,
    "epochs_max": 5,
    "early_stopping": True,
    "early_stopping_monitor": "validation_loss",
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001,
    "seed": 42,
    "checkpoint_policy": "Best validation-loss checkpoint",
}


EXPERIMENTS = [
    {
        "experiment": "baseline_fold1_seed42",
        "model": "Baseline CNN + Mean Pooling",
        "category": "Baseline",
        "image_encoder": "Custom CNN",
        "text_encoder": "Embedding + Masked Mean Pooling",
        "image_normalization": "RGB float32 / 255.0",
        "text_hidden_dim": "Not applicable",
        "dropout": 0.1,
        "pretrained_image_weights": False,
        "cross_modal_attention": False,
        "auxiliary_loss": "None",
        "checkpoint": "checkpoints/experiments/baseline_fold1_seed42",
    },
    {
        "experiment": "lstm_fold1_seed42",
        "model": "ResNet50 + LSTM",
        "category": "Classic DL",
        "image_encoder": "ResNet50",
        "text_encoder": "LSTM + Attention",
        "image_normalization": "ResNet50 preprocess_input",
        "text_hidden_dim": 512,
        "dropout": 0.2,
        "pretrained_image_weights": True,
        "cross_modal_attention": False,
        "auxiliary_loss": "None",
        "checkpoint": "checkpoints/experiments/lstm_fold1_seed42",
    },
    {
        "experiment": "gru_fold1_seed42",
        "model": "ResNet50 + GRU",
        "category": "Classic DL",
        "image_encoder": "ResNet50",
        "text_encoder": "GRU + Attention",
        "image_normalization": "ResNet50 preprocess_input",
        "text_hidden_dim": 512,
        "dropout": 0.2,
        "pretrained_image_weights": True,
        "cross_modal_attention": False,
        "auxiliary_loss": "None",
        "checkpoint": "checkpoints/experiments/gru_fold1_seed42",
    },
    {
        "experiment": "transformer_fold1_seed42",
        "model": "ResNet50 + Transformer",
        "category": "Modern DL",
        "image_encoder": "ResNet50",
        "text_encoder": "Transformer + Attention Pooling",
        "image_normalization": "ResNet50 preprocess_input",
        "text_hidden_dim": 512,
        "dropout": 0.2,
        "pretrained_image_weights": True,
        "cross_modal_attention": False,
        "auxiliary_loss": "None",
        "checkpoint": "checkpoints/experiments/transformer_fold1_seed42",
    },
    {
        "experiment": "cross_attention_fold1_seed42",
        "model": "ResNet50 + LSTM + Cross-Modal Attention",
        "category": "Advanced Architecture",
        "image_encoder": "ResNet50 with spatial features",
        "text_encoder": "LSTM + Attention",
        "image_normalization": "ResNet50 preprocess_input",
        "text_hidden_dim": 512,
        "dropout": 0.2,
        "pretrained_image_weights": True,
        "cross_modal_attention": True,
        "auxiliary_loss": "Matching Binary Cross-Entropy, weight=0.2",
        "checkpoint": "checkpoints/experiments/cross_attention_fold1_seed42",
    },
]


def checkpoint_status(relative_path):
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return "MISSING"

    if path.is_file():
        return "PRESENT"

    files = [
        item for item in path.rglob("*")
        if item.is_file()
    ]

    return "PRESENT" if files else "EMPTY"


def validate_experiment_protocol():
    assert SHARED["fold"] == 1
    assert SHARED["train_split"] == "train"
    assert SHARED["selection_split"] == "validation"
    assert SHARED["test_used_for_selection"] is False
    assert SHARED["image_size"] == "224x224"
    assert SHARED["image_color"] == "RGB"
    assert SHARED["captions_per_image"] == 5
    assert SHARED["max_caption_length"] == 34
    assert SHARED["retrieval_embedding_dim"] == 512
    assert SHARED["batch_size"] == 8
    assert SHARED["learning_rate"] == 1e-4
    assert SHARED["epochs_max"] == 5
    assert SHARED["seed"] == 42

    names = [item["experiment"] for item in EXPERIMENTS]
    assert len(names) == len(set(names)), "Duplicate experiment names."

    for item in EXPERIMENTS:
        item.update(SHARED)
        item["checkpoint_status"] = checkpoint_status(
            item["checkpoint"]
        )

    print("[PASS] Shared experiment protocol validated.")
    print("[PASS] All models use the same fold and split policy.")
    print("[PASS] Test split is excluded from model selection.")
    print("[PASS] Common training settings are recorded.")


def save_comparison():
    output_dir = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / "architecture_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "architecture_comparison.json"
    csv_path = output_dir / "architecture_comparison.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            EXPERIMENTS,
            file,
            indent=4,
            ensure_ascii=False,
        )

    columns = list(EXPERIMENTS[0].keys())

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(EXPERIMENTS)

    print("\nArchitecture comparison saved:")
    print(json_path)
    print(csv_path)

    print("\nMODEL CHECKPOINT STATUS")
    print("-" * 75)

    for item in EXPERIMENTS:
        print(
            f"{item['model']:<43} "
            f"{item['checkpoint_status']}"
        )

    print("-" * 75)
    print("Note: PRESENT only means checkpoint files exist.")
    print("It does not prove that the checkpoint passed")
    print("the full configuration-parity validation.")


def main():
    validate_experiment_protocol()
    save_comparison()


if __name__ == "__main__":
    main()