
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

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    if not isinstance(vocabulary, dict):
        raise TypeError("Vocabulary must be a dictionary.")

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

    # Build variables before restoring checkpoint weights.
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



def resolve_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    # Prefer the best validation checkpoint.
    best_dir = checkpoint_dir / "best"

    if best_dir.is_dir():
        best_path = tf.train.latest_checkpoint(
            str(best_dir)
        )

        if best_path is not None:
            return best_path

    # Fall back to the latest training checkpoint.
    checkpoint_path = tf.train.latest_checkpoint(
        str(checkpoint_dir)
    )

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No TensorFlow checkpoint found in "
            f"{checkpoint_dir}"
        )

    return checkpoint_path



def restore_model(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(model=model)

    status = checkpoint.restore(checkpoint_path)

    # Optimizer and training-state values may exist in the
    # training checkpoint but are not needed for evaluation.
    status.expect_partial()

    return model


def load_checkpoint_metadata(checkpoint_dir):
    """
    Read optional metadata if the checkpoint manager saved it.

    Metadata is optional because the evaluator must not depend
    on a particular JSON filename or training logger format.
    """

    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    possible_names = [
        "metadata.json",
        "checkpoint_metadata.json",
        "training_state.json",
    ]

    for filename in possible_names:
        metadata_path = checkpoint_dir / filename

        if metadata_path.is_file():
            try:
                with metadata_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    data = json.load(file)

                if isinstance(data, dict):
                    return data

            except (json.JSONDecodeError, OSError):
                print(
                    f"Warning: Could not read {metadata_path}"
                )

    return {}


def save_results(
    results,
    config,
    checkpoint_path,
    checkpoint_metadata,
):
    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_results = {
        key: value
        for key, value in results.items()
        if key not in {"image_names", "names"}
    }

    serializable_results = {}

    for key, value in clean_results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                metric: (
                    float(metric_value)
                    if isinstance(
                        metric_value,
                        (int, float),
                    )
                    else metric_value
                )
                for metric, metric_value in value.items()
            }
        elif isinstance(value, (int, float)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value

    serializable_results.update({
        "model_type": config["model_name"],
        "checkpoint_path": str(checkpoint_path),
        "split": "validation",
        "fold": 1,
        "seed": 42,
        "test_set_used": False,
        "checkpoint_metadata": checkpoint_metadata,
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

    print(f"\nResults saved to: {output_path}")


def run_evaluation(experiment_name):
    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}"
        )

    config = EXPERIMENTS[experiment_name]

    print("=" * 65)
    print(
        f"VISIONMATCH AI - {config['model_name']} EVALUATION"
    )
    print("=" * 65)

    print("\nLoading vocabulary...")
    vocabulary = load_vocabulary()

    print("Vocabulary size:", len(vocabulary))

    print("\nLoading validation dataset...")

    validation_data = Flickr30kDataset(
        fold_number=1,
        split="validation",
    )

    validation_dataset = validation_data.as_dataset(
        batch_size=32,
        shuffle=False,
        prefetch=True,
    )

    print("Validation images:", len(validation_data))

    print("\nBuilding TensorFlow model...")

    model = build_model(
        vocab_size=len(vocabulary),
        config=config,
    )

    print("\nResolving checkpoint...")

    checkpoint_path = resolve_checkpoint(
        config["checkpoint_dir"]
    )

    print("Checkpoint:", checkpoint_path)

    print("\nRestoring model weights...")

    model = restore_model(
        model,
        checkpoint_path,
    )

    checkpoint_metadata = load_checkpoint_metadata(
        config["checkpoint_dir"]
    )

    print("\nInitializing Retrieval Evaluator...")

    evaluator = RetrievalEvaluator(
        model=model,
        query_batch_size=32,
        candidate_chunk_size=8192,
        ks=(1, 5, 10),
    )

    print("\nEvaluating on validation split...")

    results = evaluator.evaluate(
        validation_dataset
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL RESULTS")
    print("=" * 65)

    for direction, metrics in results.items():
        if isinstance(metrics, dict):
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
