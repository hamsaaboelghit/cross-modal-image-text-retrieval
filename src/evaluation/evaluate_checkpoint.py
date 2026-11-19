"""Evaluate an existing VisionMatch checkpoint on the validation split only.

The evaluator does not discover or assume that all model runs have finished.
Pass the checkpoint for one completed run explicitly. It reads the fold and
training metadata from that checkpoint, then evaluates both retrieval
directions without touching the held-out test split.
"""

import argparse
import json
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
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.matching_evaluator import MatchingEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOCABULARY_DIR = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k" / "vocabularies"
)


def resolve_checkpoint(checkpoint_arg: str, kind: str) -> str:
    """Resolve an experiment directory, best/latest directory, or prefix."""
    path = Path(checkpoint_arg).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # TensorFlow checkpoint prefixes have a sibling '<prefix>.index' file.
    # Check this before Path.is_file(), because the prefix itself is not a file.
    prefix_candidate = str(path)
    if Path(prefix_candidate + ".index").is_file():
        return prefix_candidate

    if path.is_file():
        if str(path).endswith(".index"):
            prefix_candidate = str(path)[:-6]
            if Path(prefix_candidate + ".index").is_file():
                return prefix_candidate
        raise FileNotFoundError(
            f"Expected a TensorFlow checkpoint prefix or .index file: {path}"
        )

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidates = [path / kind, path]
    for candidate in candidates:
        if candidate.is_dir():
            prefix = tf.train.latest_checkpoint(str(candidate))
            if prefix:
                return prefix

    raise FileNotFoundError(
        f"No {kind} TensorFlow checkpoint found under: {path}"
    )


def read_checkpoint_metadata(prefix: str) -> dict:
    """Read the sidecar written by CheckpointManager for this exact prefix."""
    metadata_path = Path(prefix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Checkpoint metadata is required but missing: "
            f"{metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("config"), dict
    ):
        raise ValueError("Checkpoint metadata must contain a config object.")

    required_fields = {
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "loss_type",
        "temperature",
        "seed",
        "fold",
    }
    missing = required_fields - metadata["config"].keys()
    if missing:
        raise ValueError(
            "Checkpoint training config is incomplete: "
            f"{sorted(missing)}"
        )

    fold = metadata["config"]["fold"]
    seed = metadata["config"]["seed"]
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(1, 6):
        raise ValueError(f"Invalid fold in checkpoint metadata: {fold!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Invalid seed in checkpoint metadata: {seed!r}")

    return metadata


def build_model(vocab_size: int, args):
    """Build the exact selected architecture before restoring its weights."""
    if args.model_type == "baseline":
        if args.cross_modal_attention:
            raise ValueError("Baseline does not support cross-modal attention.")
        return BaselineRetrievalModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
        )

    if args.encoder_type is None:
        raise ValueError("--encoder-type is required when --model-type full.")

    return CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type=args.encoder_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        use_attention=args.use_attention,
        pretrained=args.pretrained,
        unfreeze_last_block=args.unfreeze_last_block,
        train_all_backbone=args.train_all_backbone,
        projection_dropout=args.projection_dropout,
        use_cross_modal_attention=args.cross_modal_attention,
        cross_modal_num_heads=args.cross_modal_num_heads,
        cross_modal_dropout=args.cross_modal_dropout,
        output_dim=args.output_dim,
    )


def restore_checkpoint(model, prefix: str):
    """Restore model weights from CheckpointManager's 'model' object."""
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(prefix)
    # CheckpointManager also stores optimizer and scalar training state.
    # Those are intentionally absent during inference; all model objects
    # must match, while extra checkpoint objects are expected.
    status.assert_existing_objects_matched()
    status.expect_partial()
    print("[PASS] Model weights restored; non-model training state ignored.")


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one completed VisionMatch checkpoint on validation only."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Experiment directory, best/latest directory, or checkpoint prefix.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest"),
        default="best",
        help="Used when --checkpoint points to an experiment directory.",
    )
    parser.add_argument(
        "--model-type", choices=("baseline", "full"), required=True
    )
    parser.add_argument(
        "--encoder-type",
        choices=("rnn", "lstm", "gru", "transformer"),
        default=None,
    )
    parser.add_argument("--cross-modal-attention", action="store_true")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--train-all-backbone", action="store_true")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument(
        "--use-attention", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--projection-dropout", type=float, default=0.1)
    parser.add_argument("--cross-modal-num-heads", type=int, default=8)
    parser.add_argument("--cross-modal-dropout", type=float, default=0.1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size; independent of the training batch size.",
    )
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--evaluate-matching",
        action="store_true",
        help="Also evaluate the matching head; requires cross-modal attention.",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    if args.model_type == "full" and args.encoder_type is None:
        parser.error("--encoder-type is required for a full model.")
    if args.evaluate_matching and not args.cross_modal_attention:
        parser.error("--evaluate-matching requires --cross-modal-attention.")
    if args.model_type == "baseline" and args.cross_modal_attention:
        parser.error("The baseline model does not support cross-modal attention.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.evaluate_matching and args.batch_size < 2:
        parser.error("Matching evaluation requires --batch-size >= 2.")
    if args.query_batch_size < 1 or args.candidate_chunk_size < 1:
        parser.error("Query and candidate chunk sizes must be positive.")
    if args.num_layers < 1 or args.embedding_dim < 1 or args.hidden_dim < 1:
        parser.error("Model dimensions and number of layers must be positive.")
    if args.output_dim < 1 or args.cross_modal_num_heads < 1:
        parser.error("Output dimension and cross-modal head count must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1).")
    if not 0.0 <= args.projection_dropout < 1.0:
        parser.error("--projection-dropout must be in [0, 1).")
    if not 0.0 <= args.cross_modal_dropout < 1.0:
        parser.error("--cross-modal-dropout must be in [0, 1).")
    return args


def main():
    args = parse_args()
    prefix = resolve_checkpoint(args.checkpoint, args.checkpoint_kind)
    metadata = read_checkpoint_metadata(prefix)
    training_config = metadata["config"]
    fold = training_config["fold"]
    seed = training_config["seed"]

    # Evaluation is deterministic and uses the fold recorded by training.
    tf.keras.utils.set_random_seed(seed)

    print("=" * 72)
    print("VISIONMATCH AI — VALIDATION CHECKPOINT EVALUATION")
    print("=" * 72)
    print("Checkpoint:", prefix)
    print("Model type:", args.model_type)
    print("Checkpoint epoch (0-based):", metadata.get("epoch"))
    print("Checkpoint validation loss:", metadata.get("val_loss"))
    print("Training fold / seed:", fold, "/", seed)
    print("Training configuration:")
    print(json.dumps(training_config, indent=2, ensure_ascii=False))
    print(
        "[NOTICE] Architecture arguments are supplied by the caller because "
        "the current checkpoint metadata does not persist all model arguments."
    )

    vocabulary_path = VOCABULARY_DIR / f"fold_{fold}_vocab.json"
    vocabulary = load_vocabulary(vocabulary_path)
    model = build_model(len(vocabulary), args)

    # Build every model variable before restoring checkpoint tensors.
    # BOS=2 and EOS=3 are valid vocabulary special-token IDs.
    dummy_images = tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)
    dummy_caption = tf.concat(
        [
            tf.constant([[2, 3]], dtype=tf.int32),
            tf.zeros((1, MAX_LENGTH - 2), dtype=tf.int32),
        ],
        axis=1,
    )
    dummy_captions = tf.tile(
        dummy_caption[:, tf.newaxis, :],
        (1, EXPECTED_CAPTIONS, 1),
    )
    model(dummy_images, dummy_captions, training=False)
    restore_checkpoint(model, prefix)

    validation_data = Flickr30kDataset(
        fold_number=fold,
        split="validation",
        caption_count=EXPECTED_CAPTIONS,
        image_augmentation="none",
    )
    validation_dataset = validation_data.as_dataset(
        batch_size=args.batch_size,
        shuffle=False,
        drop_remainder=False,
        prefetch=False,
        seed=seed,
    )

    retrieval_evaluator = RetrievalEvaluator(
        model=model,
        query_batch_size=args.query_batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        ks=(1, 5, 10),
    )
    retrieval_results = retrieval_evaluator.evaluate(validation_dataset)
    retrieval_results.pop("image_names", None)

    architecture = {
        "model_type": args.model_type,
        "encoder_type": args.encoder_type if args.model_type == "full" else None,
        "cross_modal_attention": args.cross_modal_attention,
        "pretrained": args.pretrained if args.model_type == "full" else False,
        "unfreeze_last_block": (
            args.unfreeze_last_block if args.model_type == "full" else False
        ),
        "train_all_backbone": (
            args.train_all_backbone if args.model_type == "full" else False
        ),
        "bidirectional": args.bidirectional if args.model_type == "full" else False,
        "use_attention": args.use_attention if args.model_type == "full" else None,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim if args.model_type == "full" else None,
        "num_layers": args.num_layers if args.model_type == "full" else None,
        "output_dim": args.output_dim,
        "dropout": args.dropout,
        "projection_dropout": (
            args.projection_dropout if args.model_type == "full" else None
        ),
        "cross_modal_num_heads": (
            args.cross_modal_num_heads if args.cross_modal_attention else None
        ),
        "cross_modal_dropout": (
            args.cross_modal_dropout if args.cross_modal_attention else None
        ),
    }

    payload = {
        "experiment": Path(prefix).parent.parent.name,
        "checkpoint_path": prefix,
        "checkpoint_metadata": metadata,
        "model_architecture_supplied_for_restore": architecture,
        "evaluation": {
            "split": "validation",
            "fold": fold,
            "training_seed": seed,
            "evaluation_batch_size": args.batch_size,
            "shuffle": False,
            "drop_remainder": False,
            "prefetch": False,
            "image_size": IMAGE_SIZE,
            "captions_per_image": EXPECTED_CAPTIONS,
            "max_length": MAX_LENGTH,
            "query_batch_size": args.query_batch_size,
            "candidate_chunk_size": args.candidate_chunk_size,
            "recall_k": [1, 5, 10],
            "test_set_used": False,
        },
        "retrieval_metrics": retrieval_results,
    }

    if args.evaluate_matching:
        matching_evaluator = MatchingEvaluator(model=model)
        matching_dataset = validation_data.as_dataset(
            batch_size=args.batch_size,
            shuffle=False,
            drop_remainder=False,
            prefetch=False,
            seed=seed,
        )
        payload["matching_metrics"] = matching_evaluator.evaluate(
            matching_dataset
        )

    default_output = (
        PROJECT_ROOT
        / "results"
        / "checkpoint_evaluations"
        / (Path(prefix).name + "_validation.json")
    )
    output_path = Path(args.output).expanduser() if args.output else default_output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    save_json(output_path, payload)

    print("\nVALIDATION RETRIEVAL RESULTS")
    for direction in ("text_to_image", "image_to_text"):
        print(direction.upper())
        for metric, value in retrieval_results[direction].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.6f}")

    if "matching_metrics" in payload:
        print("\nMATCHING RESULTS")
        for metric, value in payload["matching_metrics"].items():
            print(f"  {metric}: {value}")

    print("\nSaved:", output_path)
    print("Held-out test split used: NO")


if __name__ == "__main__":
    main()
