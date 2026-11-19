"""Evaluate a completed VisionMatch checkpoint without assuming a fixed run budget.

This runner is intentionally separate from evaluation_runner.py:
- accepts timestamped checkpoint directories or checkpoint prefixes;
- reads and preserves the exact TrainingConfig stored beside the checkpoint;
- requires architecture settings explicitly because current checkpoint metadata
  stores training settings but not the full model architecture;
- evaluates validation retrieval in both directions;
- optionally evaluates the cross-modal matching head separately.

It never evaluates the held-out test split.
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
FOLD = 1
SEED = 42
VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / f"fold_{FOLD}_vocab.json"
)


def resolve_checkpoint(checkpoint_arg: str, kind: str) -> str:
    path = Path(checkpoint_arg).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.is_file():
        prefix = str(path)
        if prefix.endswith(".index"):
            prefix = prefix[:-6]
        if not Path(prefix + ".index").is_file():
            raise FileNotFoundError(f"Invalid checkpoint prefix: {prefix}")
        return prefix

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
    metadata_path = Path(prefix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata is required but missing: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not isinstance(metadata, dict) or not isinstance(metadata.get("config"), dict):
        raise ValueError("Checkpoint metadata must contain a config object.")

    required_training_fields = {
        "epochs", "batch_size", "learning_rate", "optimizer",
        "loss_type", "temperature", "seed", "fold",
    }
    missing = required_training_fields - metadata["config"].keys()
    if missing:
        raise ValueError(
            f"Checkpoint training config is incomplete: {sorted(missing)}"
        )

    if int(metadata["config"]["fold"]) != FOLD:
        raise ValueError(
            f"This runner is configured for fold {FOLD}, but checkpoint "
            f"metadata says fold {metadata['config']['fold']}."
        )

    if int(metadata["config"]["seed"]) != SEED:
        print(
            "[WARNING] Checkpoint seed differs from the default project seed: "
            f"{metadata['config']['seed']}. Recording the checkpoint value."
        )

    return metadata


def build_model(vocab_size: int, args):
    if args.model_type == "baseline":
        return BaselineRetrievalModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
        )

    return CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type=args.encoder_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=False,
        use_attention=True,
        pretrained=args.pretrained,
        unfreeze_last_block=args.unfreeze_last_block,
        train_all_backbone=False,
        use_cross_modal_attention=args.cross_modal_attention,
        output_dim=args.output_dim,
    )


def restore_checkpoint(model, prefix: str):
    status = tf.train.Checkpoint(model=model).restore(prefix)
    status.assert_existing_objects_matched()
    status.expect_partial()
    print("[PASS] Checkpoint weights restored.")


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a completed VisionMatch checkpoint on validation only."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Timestamped experiment directory or TensorFlow checkpoint prefix.")
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"), default="best")
    parser.add_argument("--model-type", choices=("baseline", "full"), required=True)
    parser.add_argument("--encoder-type",
                        choices=("rnn", "lstm", "gru", "transformer"), default="lstm")
    parser.add_argument("--cross-modal-attention", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--evaluate-matching", action="store_true",
                        help="Also evaluate the matching head; requires --cross-modal-attention.")
    parser.add_argument("--output", default=None,
                        help="Optional output JSON path.")
    args = parser.parse_args()

    if args.evaluate_matching and not args.cross_modal_attention:
        parser.error("--evaluate-matching requires --cross-modal-attention.")

    if args.model_type == "baseline" and args.cross_modal_attention:
        parser.error("The baseline model does not support cross-modal attention.")

    if args.batch_size < 2 and args.evaluate_matching:
        parser.error("Matching evaluation requires batch size >= 2.")

    tf.keras.utils.set_random_seed(SEED)

    prefix = resolve_checkpoint(args.checkpoint, args.checkpoint_kind)
    metadata = read_checkpoint_metadata(prefix)
    training_config = metadata["config"]

    print("=" * 68)
    print("VISIONMATCH AI — CHECKPOINT EVALUATION")
    print("=" * 68)
    print("Checkpoint:", prefix)
    print("Checkpoint epoch (0-based):", metadata.get("epoch"))
    print("Checkpoint validation loss:", metadata.get("val_loss"))
    print("Recorded training config:")
    print(json.dumps(training_config, indent=2, ensure_ascii=False))

    vocabulary = load_vocabulary(VOCAB_PATH)
    model = build_model(len(vocabulary), args)

    dummy_images = tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32)
    dummy_captions = tf.fill(
        (1, EXPECTED_CAPTIONS, MAX_LENGTH),
        tf.constant(4, dtype=tf.int32),
    )
    model(dummy_images, dummy_captions, training=False)
    restore_checkpoint(model, prefix)

    validation_data = Flickr30kDataset(
        fold_number=FOLD,
        split="validation",
    )
    validation_dataset = validation_data.as_dataset(
        batch_size=args.batch_size,
        shuffle=False,
        drop_remainder=False,
        prefetch=True,
        seed=int(training_config["seed"]),
    )

    retrieval_evaluator = RetrievalEvaluator(
        model=model,
        query_batch_size=args.query_batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        ks=(1, 5, 10),
    )
    retrieval_results = retrieval_evaluator.evaluate(validation_dataset)
    retrieval_results.pop("image_names", None)

    payload = {
        "experiment": Path(prefix).parent.parent.name,
        "checkpoint_path": prefix,
        "checkpoint_metadata": metadata,
        "model_architecture": {
            "model_type": args.model_type,
            "encoder_type": None if args.model_type == "baseline" else args.encoder_type,
            "cross_modal_attention": args.cross_modal_attention,
            "pretrained": args.pretrained if args.model_type == "full" else False,
            "unfreeze_last_block": args.unfreeze_last_block if args.model_type == "full" else False,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim if args.model_type == "full" else None,
            "num_layers": args.num_layers if args.model_type == "full" else None,
            "output_dim": args.output_dim,
            "dropout": args.dropout,
        },
        "evaluation": {
            "split": "validation",
            "fold": FOLD,
            "batch_size": args.batch_size,
            "shuffle": False,
            "drop_remainder": False,
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
        # A fresh deterministic dataset iterator is created from the same
        # validation split; no weights or checkpoints are modified.
        matching_dataset = validation_data.as_dataset(
            batch_size=args.batch_size,
            shuffle=False,
            drop_remainder=False,
            prefetch=True,
            seed=int(training_config["seed"]),
        )
        payload["matching_metrics"] = matching_evaluator.evaluate(matching_dataset)

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
