
import argparse
import json
import math
from pathlib import Path

import tensorflow as tf

from src.data.flickr30k_dataset import (
    Flickr30kDataset,
    IMAGE_SIZE,
    EXPECTED_CAPTIONS,
    MAX_LENGTH,
)
from src.data.vocabulary import load_vocabulary
from src.models.baseline_model import BaselineRetrievalModel
from src.models.full_model import CrossModalRetrievalModel
from src.training.train_config import TrainingConfig
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOLD = 1
SEED = 42

BATCH_SIZE = 8
EMBEDDING_DIM = 512

QUERY_BATCH_SIZE = 32
CANDIDATE_CHUNK_SIZE = 8192
RETRIEVAL_KS = (1, 5, 10)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / f"fold_{FOLD}_vocab.json"
)


EXPERIMENTS = {
    "baseline": {
        "model_type": "baseline",
        "model_name": "CNN-Baseline",
        "encoder_type": None,
        "use_cross_modal_attention": False,
        "pretrained": False,
        "unfreeze_last_block": False,
        "embedding_dim": 300,
        "output_dim": 512,
        "dropout": 0.1,
        "checkpoint_dir": (
            "checkpoints/experiments/baseline_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/baseline_fold1_seed42"
        ),
    },
    "gru": {
        "model_type": "full",
        "model_name": "ResNet50-GRU",
        "encoder_type": "gru",
        "use_cross_modal_attention": False,
        "pretrained": True,
        "unfreeze_last_block": False,
        "embedding_dim": 300,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "checkpoint_dir": (
            "checkpoints/experiments/gru_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/gru_fold1_seed42"
        ),
    },
    "transformer": {
        "model_type": "full",
        "model_name": "ResNet50-Transformer",
        "encoder_type": "transformer",
        "use_cross_modal_attention": False,
        "pretrained": True,
        "unfreeze_last_block": False,
        "embedding_dim": 300,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "checkpoint_dir": (
            "checkpoints/experiments/transformer_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/transformer_fold1_seed42"
        ),
    },
    "cross_attention": {
        "model_type": "full",
        "model_name": "ResNet50-Cross-Modal-Attention",
        "encoder_type": "lstm",
        "use_cross_modal_attention": True,
        "pretrained": True,
        "unfreeze_last_block": False,
        "embedding_dim": 300,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "checkpoint_dir": (
            "checkpoints/experiments/cross_attention_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/cross_attention_fold1_seed42"
        ),
    },
}


def set_reproducibility():
    tf.keras.utils.set_random_seed(SEED)


def validate_shared_configuration():
    expected = {
        "image_size": 224,
        "captions_per_image": 5,
        "max_length": 34,
        "embedding_dim": 512,
        "fold": 1,
        "seed": 42,
        "batch_size": 8,
    }

    actual = {
        "image_size": IMAGE_SIZE,
        "captions_per_image": EXPECTED_CAPTIONS,
        "max_length": MAX_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "fold": FOLD,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
    }

    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise ValueError(
                f"Configuration mismatch for {key}: "
                f"expected {expected_value}, found {actual[key]}."
            )

        print(f"[PASS] {key}: {actual[key]}")


def load_project_vocabulary():
    vocabulary = load_vocabulary(VOCAB_PATH)

    print("[PASS] Vocabulary loaded and validated.")
    print("Vocabulary size:", len(vocabulary))

    return vocabulary


def build_expected_training_config(experiment_name):
    return TrainingConfig(
        epochs=5,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="none",
        loss_type="info_nce",
        temperature=0.07,
        seed=SEED,
        fold=FOLD,
        checkpoint_dir=EXPERIMENTS[
            experiment_name
        ]["checkpoint_dir"],
        log_dir="runs",
        save_best_only=True,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
    ).to_dict()


def validate_training_config(metadata, experiment_name):
    if "config" not in metadata:
        raise ValueError(
            "Checkpoint metadata does not contain training config."
        )

    saved_config = metadata["config"]

    if not isinstance(saved_config, dict):
        raise TypeError(
            "Checkpoint training config must be a dictionary."
        )

    expected_config = build_expected_training_config(
        experiment_name
    )

    # Older checkpoints may not contain this field.
    # It is not used by the current InfoNCE training pipeline.
    legacy_optional_fields = {"matching_loss_weight"}

    mismatches = []
    unrecorded_fields = []

    for key, expected_value in expected_config.items():
        if key not in saved_config:
            if key in legacy_optional_fields:
                unrecorded_fields.append(key)
                continue

            mismatches.append(
                f"{key}: missing from checkpoint metadata"
            )
            continue

        actual_value = saved_config[key]

        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value

        if not matches:
            mismatches.append(
                f"{key}: expected {expected_value}, "
                f"found {actual_value}"
            )

    if mismatches:
        details = "\n".join(
            f"- {item}" for item in mismatches
        )

        raise ValueError(
            "Training/Evaluation configuration mismatch:\n"
            f"{details}"
        )

    print(
        "[PASS] All recorded training configuration fields match."
    )
    print("[PASS] Epochs, batch size and learning rate match.")
    print("[PASS] Optimizer and scheduler match.")
    print("[PASS] Loss configuration matches.")
    print("[PASS] Fold and seed match.")
    print("[PASS] Checkpoint directory matches.")

    for key in unrecorded_fields:
        print(
            f"[NOT RECORDED] {key}: "
            "not present in this checkpoint's metadata; "
            "it was not used by the current training loss."
        )

    if unrecorded_fields:
        print(
            "[NOTE] Configuration parity is verified for all "
            "recorded fields; unrecorded legacy fields are "
            "not claimed as verified."
        )


def validate_architecture_config(config):
    if config["model_type"] == "baseline":
        expected = {
            "embedding_dim": 300,
            "output_dim": 512,
            "dropout": 0.1,
        }

        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(
                    f"Baseline architecture mismatch: {key}"
                )

    elif config["model_type"] == "full":
        expected = {
            "embedding_dim": 300,
            "hidden_dim": 512,
            "num_layers": 1,
            "dropout": 0.2,
            "bidirectional": False,
            "use_attention": True,
            "pretrained": True,
            "unfreeze_last_block": False,
        }

        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(
                    f"Full model architecture mismatch: {key}"
                )

    else:
        raise ValueError(
            f"Unsupported model type: {config['model_type']}"
        )

    print("[PASS] Architecture configuration matches training.")

    if config["model_type"] == "full":
        print("[PASS] Encoder type:", config["encoder_type"])
        print(
            "[PASS] Cross-modal attention:",
            config["use_cross_modal_attention"],
        )


def build_model(vocab_size, config):
    if config["model_type"] == "baseline":
        model = BaselineRetrievalModel(
            vocab_size=vocab_size,
            embedding_dim=config["embedding_dim"],
            output_dim=config["output_dim"],
            dropout=config["dropout"],
        )

    else:
        model = CrossModalRetrievalModel(
            vocab_size=vocab_size,
            encoder_type=config["encoder_type"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            bidirectional=config["bidirectional"],
            use_attention=config["use_attention"],
            pretrained=config["pretrained"],
            unfreeze_last_block=config["unfreeze_last_block"],
            use_cross_modal_attention=config[
                "use_cross_modal_attention"
            ],
        )

    dummy_images = tf.zeros(
        (1, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=tf.float32,
    )

    dummy_captions = tf.fill(
        (1, EXPECTED_CAPTIONS, MAX_LENGTH),
        tf.constant(4, dtype=tf.int32),
    )

    model(
        dummy_images,
        dummy_captions,
        training=False,
    )

    return model


def resolve_checkpoint(
    checkpoint_dir,
    checkpoint_kind="best",
):
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {checkpoint_dir}"
        )

    if checkpoint_dir.is_file():
        checkpoint_path = str(checkpoint_dir)

        if checkpoint_path.endswith(".index"):
            checkpoint_path = checkpoint_path[:-6]

        if not Path(checkpoint_path + ".index").is_file():
            raise FileNotFoundError(
                f"Invalid TensorFlow checkpoint: {checkpoint_path}"
            )

        return checkpoint_path

    direct_checkpoint = tf.train.latest_checkpoint(
        str(checkpoint_dir)
    )

    if direct_checkpoint is not None:
        return direct_checkpoint

    if checkpoint_kind not in {"best", "latest"}:
        raise ValueError(
            "checkpoint_kind must be 'best' or 'latest'."
        )

    preferred = [
        checkpoint_kind,
        "latest" if checkpoint_kind == "best" else "best",
    ]

    for kind in preferred:
        candidate_dir = checkpoint_dir / kind

        if not candidate_dir.is_dir():
            continue

        checkpoint_path = tf.train.latest_checkpoint(
            str(candidate_dir)
        )

        if checkpoint_path is not None:
            return checkpoint_path

    raise FileNotFoundError(
        f"No TensorFlow checkpoint found in {checkpoint_dir}."
    )


def load_checkpoint_metadata(checkpoint_path):
    metadata_path = Path(
        str(checkpoint_path) + ".json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            "Checkpoint metadata must be a JSON object."
        )

    return metadata


def restore_model(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(
        model=model,
    )

    status = checkpoint.restore(checkpoint_path)

    status.assert_existing_objects_matched()
    status.expect_partial()

    print("[PASS] Model weights restored.")

    return model


def build_validation_dataset():
    validation_data = Flickr30kDataset(
        fold_number=FOLD,
        split="validation",
    )

    validation_dataset = validation_data.as_dataset(
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
        seed=SEED,
    )

    print("Validation images:", len(validation_data))
    print("Batch size:", BATCH_SIZE)
    print("Shuffle:", False)
    print("Drop remainder:", False)

    return validation_data, validation_dataset


def to_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_safe(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return str(value)


def save_results(
    results,
    experiment_name,
    config,
    checkpoint_path,
    checkpoint_metadata,
):
    output_dir = (
        PROJECT_ROOT / config["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_results = {
        key: value
        for key, value in results.items()
        if key not in {"image_names", "names"}
    }

    serializable_results = to_json_safe(
        clean_results
    )

    serializable_results.update({
        "experiment": experiment_name,
        "model_type": config["model_name"],
        "encoder_type": config["encoder_type"],
        "use_cross_modal_attention": config[
            "use_cross_modal_attention"
        ],
        "architecture_config": {
            key: value
            for key, value in config.items()
            if key not in {"checkpoint_dir", "output_dir"}
        },
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata": to_json_safe(
            checkpoint_metadata
        ),
        "evaluation_config": {
            "fold": FOLD,
            "seed": SEED,
            "split": "validation",
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "max_length": MAX_LENGTH,
            "captions_per_image": EXPECTED_CAPTIONS,
            "embedding_dim": EMBEDDING_DIM,
            "query_batch_size": QUERY_BATCH_SIZE,
            "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
            "recall_k": list(RETRIEVAL_KS),
            "shuffle": False,
            "drop_remainder": False,
        },
        "test_set_used": False,
    })

    output_path = (
        output_dir / "retrieval_metrics.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nResults saved to:", output_path)

    return output_path


def run_evaluation(
    experiment_name,
    checkpoint_kind="best",
):
    if experiment_name not in EXPERIMENTS:
        available = ", ".join(
            sorted(EXPERIMENTS.keys())
        )

        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {available}"
        )

    config = EXPERIMENTS[experiment_name]

    print("\n" + "=" * 65)
    print(
        f"VISIONMATCH AI - {config['model_name']} EVALUATION"
    )
    print("=" * 65)

    set_reproducibility()

    print("\nChecking shared configuration...")
    validate_shared_configuration()

    print("\nChecking architecture configuration...")
    validate_architecture_config(config)

    print("\nLoading vocabulary...")
    vocabulary = load_project_vocabulary()

    print("\nResolving checkpoint...")
    checkpoint_path = resolve_checkpoint(
        config["checkpoint_dir"],
        checkpoint_kind=checkpoint_kind,
    )

    print("Checkpoint:", checkpoint_path)

    print("\nLoading checkpoint metadata...")
    checkpoint_metadata = load_checkpoint_metadata(
        checkpoint_path
    )

    print("\nValidating training/evaluation parity...")
    validate_training_config(
        metadata=checkpoint_metadata,
        experiment_name=experiment_name,
    )

    print("\nLoading validation dataset...")
    validation_data, validation_dataset = (
        build_validation_dataset()
    )

    print("\nBuilding TensorFlow model...")
    model = build_model(
        vocab_size=len(vocabulary),
        config=config,
    )

    print("\nRestoring trained weights...")
    model = restore_model(
        model,
        checkpoint_path,
    )

    print("\nInitializing Retrieval Evaluator...")
    evaluator = RetrievalEvaluator(
        model=model,
        query_batch_size=QUERY_BATCH_SIZE,
        candidate_chunk_size=CANDIDATE_CHUNK_SIZE,
        ks=RETRIEVAL_KS,
    )

    print("\nEvaluating on validation split...")
    results = evaluator.evaluate(
        validation_dataset
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL RESULTS")
    print("=" * 65)

    for direction in (
        "text_to_image",
        "image_to_text",
    ):
        metrics = results.get(direction, {})

        print(
            f"\n{direction.upper().replace('_', ' ')}"
        )

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(
                    f"  {metric_name}: {value:.6f}"
                )

    save_results(
        results=results,
        experiment_name=experiment_name,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
    )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)
    print("Training/Evaluation parity: PASS")
    print("Architecture configuration: PASS")
    print("Validation split only: PASS")
    print("Test split used: NO")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a VisionMatch image-text retrieval experiment."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=sorted(EXPERIMENTS.keys()),
    )

    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        default="best",
        choices=["best", "latest"],
    )

    args = parser.parse_args()

    run_evaluation(
        experiment_name=args.experiment,
        checkpoint_kind=args.checkpoint_kind,
    )
