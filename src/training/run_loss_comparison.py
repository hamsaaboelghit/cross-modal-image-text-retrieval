"""Controlled comparison of retrieval loss functions on a fixed baseline model."""

import argparse
import csv
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from src.data.vocabulary import load_vocabulary
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.models.baseline_model import BaselineRetrievalModel
from src.training.experiment_runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLD, SEED = 1, 42
BATCH_SIZE, LEARNING_RATE = 8, 1e-4
OUTPUT_DIM, DROPOUT, TEMPERATURE = 512, 0.1, 0.07
EVAL_BATCH_SIZE = 8
LOSS_EXPERIMENTS = [
    {"name": "info_nce", "loss_type": "info_nce", "margin": None},
    {"name": "margin_contrastive", "loss_type": "margin", "margin": 1.0},
    {"name": "triplet", "loss_type": "triplet", "margin": 0.2},
]
VOCAB_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flickr30k"
    / "vocabularies" / f"fold_{FOLD}_vocab.json"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "loss_comparison"


def atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
        file.flush()
    temporary.replace(path)


def save_results(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUTPUT_DIR / "loss_comparison_results.json", rows)
    csv_path = OUTPUT_DIR / "loss_comparison_results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    columns = [
        "experiment", "loss_type", "margin", "status", "best_epoch",
        "best_train_loss", "best_validation_loss", "final_train_loss",
        "final_validation_loss", "training_time_minutes", "checkpoint_path",
        "text_to_image_r1", "text_to_image_r5", "text_to_image_r10",
        "text_to_image_mean_recall", "text_to_image_mrr",
        "image_to_text_r1", "image_to_text_r5", "image_to_text_r10",
        "image_to_text_mean_recall", "image_to_text_mrr",
        "overall_mean_recall", "overall_mrr", "evaluation_split",
        "test_set_used", "error",
    ]
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
    temporary.replace(csv_path)


def find_best_checkpoint(experiment_name):
    directory = (
        PROJECT_ROOT / "checkpoints" / "experiments"
        / experiment_name / "best"
    )
    checkpoint = tf.train.latest_checkpoint(str(directory))
    if checkpoint is None:
        raise FileNotFoundError(f"No best checkpoint found in {directory}")
    return checkpoint


def evaluate_validation(experiment_name):
    vocabulary = load_vocabulary(VOCAB_PATH)
    model = BaselineRetrievalModel(
        vocab_size=len(vocabulary),
        embedding_dim=300,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    )
    model(
        tf.zeros((1, 224, 224, 3), dtype=tf.float32),
        tf.fill((1, 5, 34), tf.constant(4, dtype=tf.int32)),
        training=False,
    )
    checkpoint_path = find_best_checkpoint(experiment_name)
    status = tf.train.Checkpoint(model=model).restore(checkpoint_path)
    status.assert_existing_objects_matched()
    status.expect_partial()

    validation_data = Flickr30kDataset(FOLD, "validation")
    validation_dataset = validation_data.as_dataset(
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_remainder=True,
        seed=SEED,
    )
    evaluator = RetrievalEvaluator(
        model=model, query_batch_size=32,
        candidate_chunk_size=8192, ks=(1, 5, 10),
    )
    results = evaluator.evaluate(validation_dataset)
    ti, it = results["text_to_image"], results["image_to_text"]
    return {
        "checkpoint_path": str(checkpoint_path),
        "text_to_image_r1": ti["recall@1"],
        "text_to_image_r5": ti["recall@5"],
        "text_to_image_r10": ti["recall@10"],
        "text_to_image_mean_recall": ti["mean_recall"],
        "text_to_image_mrr": ti["mrr"],
        "image_to_text_r1": it["recall@1"],
        "image_to_text_r5": it["recall@5"],
        "image_to_text_r10": it["recall@10"],
        "image_to_text_mean_recall": it["mean_recall"],
        "image_to_text_mrr": it["mrr"],
        "overall_mean_recall": (ti["mean_recall"] + it["mean_recall"]) / 2.0,
        "overall_mrr": (ti["mrr"] + it["mrr"]) / 2.0,
        "evaluation_split": "validation",
        "test_set_used": False,
        "validation_images_evaluated": results["num_images"],
        "validation_embedding_dim": results["embedding_dim"],
    }


def run_comparison(experiments, epochs, resume=False, overwrite=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": "BaselineRetrievalModel", "fold": FOLD, "seed": SEED,
        "epochs": epochs, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT, "temperature": TEMPERATURE,
        "drop_last": True, "selection_split": "validation",
        "experiments": experiments,
    }
    metadata_path = OUTPUT_DIR / "loss_comparison_metadata.json"
    results_path = OUTPUT_DIR / "loss_comparison_results.json"
    if resume:
        if not metadata_path.is_file() or not results_path.is_file():
            raise FileNotFoundError("Cannot resume: metadata/results are missing.")
        with metadata_path.open("r", encoding="utf-8") as file:
            if json.load(file) != metadata:
                raise ValueError("Run settings differ; use --overwrite for a new run.")
        with results_path.open("r", encoding="utf-8") as file:
            rows = json.load(file)
        if not isinstance(rows, list):
            raise ValueError("Results file must contain a JSON list.")
    else:
        if (metadata_path.exists() or results_path.exists()) and not overwrite:
            raise FileExistsError("Results exist. Use --resume or --overwrite.")
        rows = []
        atomic_write_json(metadata_path, metadata)

    completed = {
        row["experiment"] for row in rows
        if row.get("status") == "completed" and "overall_mean_recall" in row
    }
    for index, experiment in enumerate(experiments, start=1):
        name = experiment["name"]
        if name in completed:
            print(f"[SKIP] {name} already completed.")
            continue
        rows = [row for row in rows if row.get("experiment") != name]
        run_name = f"loss_comparison_{name}"
        start = time.time()
        print(f"\nLOSS EXPERIMENT {index}/{len(experiments)}: {name}")
        try:
            history = run_experiment(
                experiment_name=run_name, model_type="baseline",
                epochs=epochs, batch_size=BATCH_SIZE, seed=SEED, fold=FOLD,
                learning_rate=LEARNING_RATE, temperature=TEMPERATURE,
                loss_type=experiment["loss_type"], margin=experiment["margin"],
                drop_last=True, output_dim=OUTPUT_DIM, dropout=DROPOUT,
                log_dir="runs/loss_comparison",
            )
            checkpoint_path = find_best_checkpoint(run_name)
            metrics = evaluate_validation(run_name)
            val_losses = history.get("val_retrieval_loss", history.get("val_loss", []))
            train_losses = history.get("train_retrieval_loss", history.get("train_loss", []))
            if not val_losses or not train_losses:
                raise RuntimeError("Training history is missing retrieval losses.")
            best_index = int(np.argmin(val_losses))
            row = {
                "experiment": name, "loss_type": experiment["loss_type"],
                "margin": experiment["margin"], "status": "completed",
                "best_epoch": best_index + 1,
                "best_train_loss": float(train_losses[best_index]),
                "best_validation_loss": float(val_losses[best_index]),
                "final_train_loss": float(train_losses[-1]),
                "final_validation_loss": float(val_losses[-1]),
                "training_time_minutes": (time.time() - start) / 60.0,
                "checkpoint_path": str(checkpoint_path), **metrics,
            }
            rows.append(row)
            print(f"[PASS] {name} | Validation Mean Recall: {row['overall_mean_recall']:.4f}")
        except tf.errors.ResourceExhaustedError as error:
            rows.append({
                "experiment": name, "loss_type": experiment["loss_type"],
                "margin": experiment["margin"], "status": "failed_out_of_memory",
                "training_time_minutes": (time.time() - start) / 60.0,
                "error": str(error)[:1000],
            })
            print(f"[OUT OF MEMORY] {name}")
        except Exception as error:
            rows.append({
                "experiment": name, "loss_type": experiment["loss_type"],
                "margin": experiment["margin"], "status": "failed",
                "training_time_minutes": (time.time() - start) / 60.0,
                "error": str(error)[:1000],
            })
            print(f"[FAILED] {name}: {error}")
            traceback.print_exc()
        finally:
            save_results(rows)
            tf.keras.backend.clear_session()
            gc.collect()

    successful = [
        row for row in rows
        if row.get("status") == "completed" and "overall_mean_recall" in row
    ]
    if successful:
        best = max(successful, key=lambda row: (
            row["overall_mean_recall"], -row["best_validation_loss"]
        ))
        atomic_write_json(OUTPUT_DIR / "best_loss_configuration.json", {
            "selection_split": "validation",
            "selection_metric": "overall_mean_recall",
            "tie_breaker": "lower_best_validation_retrieval_loss",
            "best_experiment": best, "test_set_used": False,
        })
        print(f"\nBest validation configuration: {best['experiment']}")
    save_results(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare InfoNCE, Margin Contrastive, and Triplet losses "
                    "using one fixed Baseline retrieval model."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--only", nargs="+", choices=[
        "info_nce", "margin_contrastive", "triplet"
    ])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined.")

    experiments = LOSS_EXPERIMENTS
    if args.only:
        experiments = [item for item in LOSS_EXPERIMENTS if item["name"] in args.only]
    print("MODEL: BaselineRetrievalModel | DATA: Flickr30k fold 1")
    print("FIXED: seed=42, batch=8, lr=1e-4, dim=512, dropout=0.1")
    print("TRAIN/VALIDATION LOSS: complete batches only (drop_last=True)")
    print("SELECTION: validation overall Mean Recall; test split untouched")
    for item in experiments:
        print(f"- {item['name']}: {item['loss_type']} (margin={item['margin']})")
    if args.list:
        print("List-only mode. No training was started.")
        return
    run_comparison(experiments, args.epochs, args.resume, args.overwrite)


if __name__ == "__main__":
    main()
