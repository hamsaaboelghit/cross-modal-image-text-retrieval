
import json
from pathlib import Path

import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.full_model import CrossModalRetrievalModel
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / "fold_1_vocab.json"
)

IMAGE_SIZE = 224
MAX_LENGTH = 34
NUM_CAPTIONS = 5
FOLD = 1
SEED = 42


EXPERIMENTS = {
    "gru": {
        "encoder_type": "gru",
        "use_cross_modal_attention": False,
        "checkpoint_dir": (
            "checkpoints/experiments/gru_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/gru_fold1_seed42"
        ),
        "model_name": "ResNet50-GRU",
    },
    "transformer": {
        "encoder_type": "transformer",
        "use_cross_modal_attention": False,
        "checkpoint_dir": (
            "checkpoints/experiments/transformer_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/transformer_fold1_seed42"
        ),
        "model_name": "ResNet50-Transformer",
    },
    "cross_attention": {
        "encoder_type": "lstm",
        "use_cross_modal_attention": True,
        "checkpoint_dir": (
            "checkpoints/experiments/cross_attention_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/cross_attention_fold1_seed42"
        ),
        "model_name": "ResNet50-Cross-Modal-Attention",
    },
    "real": {
        "encoder_type": "lstm",
        "use_cross_modal_attention": False,
        "checkpoint_dir": "checkpoints/real_fold1",
        "output_dir": "results/real_fold1",
        "model_name": "ResNet50-LSTM",
    },
}


def load_vocabulary():
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found: {VOCAB_PATH}"
        )

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        raise TypeError(
            "Vocabulary must be a dictionary."
        )

    required_tokens = {
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    }

    missing = required_tokens - vocabulary.keys()

    if missing:
        raise ValueError(
            f"Vocabulary is missing special tokens: {missing}"
        )

    return vocabulary


def build_model(vocab_size, config):
    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type=config["encoder_type"],
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=False,
        unfreeze_last_block=False,
        use_cross_modal_attention=config[
            "use_cross_modal_attention"
        ],
    )

    # Build all variables before restoring the checkpoint.
    dummy_images = tf.zeros(
        (1, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=tf.float32,
    )

    dummy_captions = tf.zeros(
        (1, NUM_CAPTIONS, MAX_LENGTH),
        dtype=tf.int32,
    )

    model(
        dummy_images,
        dummy_captions,
        training=False,
    )

    return model


def resolve_checkpoint(checkpoint_dir, checkpoint_kind="best"):
    """
    Resolve a checkpoint from:
      - experiment root containing best/ and latest/
      - direct best/ or latest/ directory
      - direct TensorFlow checkpoint prefix
    """

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
                f"Invalid TensorFlow checkpoint prefix: {checkpoint_path}"
            )

        return checkpoint_path

    # If the supplied path is itself a checkpoint directory.
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
        "No TensorFlow checkpoint found.\n"
        f"Experiment directory: {checkpoint_dir}\n"
        f"Checked directories: {preferred}"
    )


def restore_model(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(
        model=model,
    )

    status = checkpoint.restore(
        checkpoint_path
    )

    # The training checkpoint may also contain optimizer and state variables.
    status.expect_partial()

    return model


def load_checkpoint_metadata(checkpoint_path):
    """
    Metadata is saved beside the checkpoint prefix:
        latest_checkpoint-5.json
        best_model-5.json
    """

    metadata_path = Path(
        str(checkpoint_path) + ".json"
    )

    if not metadata_path.is_file():
        print(
            "Warning: No metadata file found for this checkpoint."
        )
        return {}

    try:
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

    except (json.JSONDecodeError, OSError) as error:
        print(
            f"Warning: Could not read checkpoint metadata: {error}"
        )
        return {}


def _to_json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def save_results(
    results,
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
        if key not in {
            "image_names",
            "names",
        }
    }

    serializable_results = _to_json_safe(
        clean_results
    )

    serializable_results.update({
        "model_type": config["model_name"],
        "encoder_type": config["encoder_type"],
        "use_cross_modal_attention": config[
            "use_cross_modal_attention"
        ],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata": _to_json_safe(
            checkpoint_metadata
        ),
        "split": "validation",
        "fold": FOLD,
        "seed": SEED,
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

    print(
        f"\nResults saved to: {output_path}"
    )

    return output_path


def run_evaluation(
    experiment_name,
    checkpoint_kind="best",
):
    if experiment_name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())

        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {available}"
        )

    config = EXPERIMENTS[experiment_name]

    print("=" * 65)
    print(
        f"VISIONMATCH AI - {config['model_name']} EVALUATION"
    )
    print("=" * 65)

    print("\nLoading vocabulary...")
    vocabulary = load_vocabulary()

    print(
        "Vocabulary size:",
        len(vocabulary),
    )

    print("\nLoading validation dataset...")

    validation_data = Flickr30kDataset(
        fold_number=FOLD,
        split="validation",
    )

    validation_dataset = validation_data.as_dataset(
        batch_size=32,
        shuffle=False,
        prefetch=True,
    )

    print(
        "Validation images:",
        len(validation_data),
    )

    print("\nBuilding TensorFlow model...")

    model = build_model(
        vocab_size=len(vocabulary),
        config=config,
    )

    print("\nResolving checkpoint...")

    checkpoint_path = resolve_checkpoint(
        config["checkpoint_dir"],
        checkpoint_kind=checkpoint_kind,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print("\nRestoring model weights...")

    model = restore_model(
        model,
        checkpoint_path,
    )

    checkpoint_metadata = load_checkpoint_metadata(
        checkpoint_path
    )

    print("\nInitializing Retrieval Evaluator...")

    evaluator = RetrievalEvaluator(
        model=model,
        query_batch_size=32,
        candidate_chunk_size=8192,
        ks=(1, 5, 10),
    )

    print(
        "\nEvaluating on validation split..."
    )

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
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
    )

    print("\nValidation evaluation completed.")
    print("Test split was NOT used.")

    return results


if __name__ == "__main__":
    import argparse

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
