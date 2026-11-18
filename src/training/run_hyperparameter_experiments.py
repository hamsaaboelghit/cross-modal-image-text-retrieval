
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
from src.models.baseline_model import BaselineRetrievalModel
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.training.experiment_runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FOLD = 1
SEED = 42
EVAL_BATCH_SIZE = 8

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / f"fold_{FOLD}_vocab.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "hyperparameter_experiments"
)


# The reference values match the existing Baseline configuration.
REFERENCE = {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "output_dim": 512,
    "dropout": 0.1,
    "temperature": 0.07,
}


def build_experiments():
    """Create one-factor-at-a-time experiments."""

    experiments = [
        {
            "name": "reference",
            "changed_parameter": "reference",
            "changed_value": "default",
            **REFERENCE,
        }
    ]

    variations = [
        ("learning_rate", 1e-3),
        ("learning_rate", 1e-5),

        ("batch_size", 16),
        ("batch_size", 32),

        ("output_dim", 256),

        ("dropout", 0.2),
        ("dropout", 0.3),

        ("temperature", 0.05),
        ("temperature", 0.1),
    ]

    for parameter, value in variations:
        config = dict(REFERENCE)
        config[parameter] = value

        experiments.append({
            "name": f"{parameter}_{str(value).replace('.', '_').replace('-', 'm')}",
            "changed_parameter": parameter,
            "changed_value": value,
            **config,
        })

    return experiments


def validate_experiment_design(experiments):
    """Verify that each experiment changes one parameter only."""

    reference_values = {
        key: REFERENCE[key]
        for key in REFERENCE
    }

    names = [item["name"] for item in experiments]

    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique.")

    if len(experiments) != 10:
        raise ValueError(
            "Expected one reference and nine variations."
        )

    for experiment in experiments:
        differences = [
            key
            for key in reference_values
            if experiment[key] != reference_values[key]
        ]

        if experiment["name"] == "reference":
            if differences:
                raise ValueError(
                    "Reference experiment must use all default values."
                )
        elif differences != [experiment["changed_parameter"]]:
            raise ValueError(
                f"{experiment['name']} changes unexpected parameters: "
                f"{differences}"
            )

    print("[PASS] One-factor-at-a-time experiment design.")
    print(f"[PASS] Total experiments: {len(experiments)}")


def find_best_checkpoint(experiment_name):
    checkpoint_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "experiments"
        / experiment_name
        / "best"
    )

    checkpoint_path = tf.train.latest_checkpoint(
        str(checkpoint_dir)
    )

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No best checkpoint found in: {checkpoint_dir}"
        )

    return checkpoint_path


def evaluate_experiment(experiment):
    """Evaluate the best checkpoint on the validation split."""

    name = experiment["name"]
    output_dim = experiment["output_dim"]

    print("\n" + "-" * 70)
    print(f"VALIDATION RETRIEVAL EVALUATION: {name}")
    print("-" * 70)

    vocabulary = load_vocabulary(VOCAB_PATH)

    model = BaselineRetrievalModel(
        vocab_size=len(vocabulary),
        embedding_dim=300,
        output_dim=output_dim,
        dropout=experiment["dropout"],
    )

    # Build model variables before restoring the checkpoint.
    dummy_images = tf.zeros(
        (1, 224, 224, 3),
        dtype=tf.float32,
    )

    dummy_captions = tf.fill(
        (1, 5, 34),
        tf.constant(4, dtype=tf.int32),
    )

    model(
        dummy_images,
        dummy_captions,
        training=False,
    )

    checkpoint_path = find_best_checkpoint(name)

    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    status.expect_partial()

    print("[PASS] Best model checkpoint restored.")

    validation_data = Flickr30kDataset(
        fold_number=FOLD,
        split="validation",
    )

    validation_dataset = validation_data.as_dataset(
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        drop_remainder=False,
        seed=SEED,
    )

    evaluator = RetrievalEvaluator(
        model=model,
        query_batch_size=32,
        candidate_chunk_size=8192,
        ks=(1, 5, 10),
    )

    results = evaluator.evaluate(validation_dataset)

    text_to_image = results["text_to_image"]
    image_to_text = results["image_to_text"]

    overall_mean_recall = (
        text_to_image["mean_recall"]
        + image_to_text["mean_recall"]
    ) / 2.0

    overall_mrr = (
        text_to_image["mrr"]
        + image_to_text["mrr"]
    ) / 2.0

    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "validation_images": results["num_images"],
        "embedding_dim": results["embedding_dim"],

        "text_to_image_r1": text_to_image["recall@1"],
        "text_to_image_r5": text_to_image["recall@5"],
        "text_to_image_r10": text_to_image["recall@10"],
        "text_to_image_mean_recall": text_to_image["mean_recall"],
        "text_to_image_mrr": text_to_image["mrr"],
        "text_to_image_median_rank": text_to_image["median_rank"],

        "image_to_text_r1": image_to_text["recall@1"],
        "image_to_text_r5": image_to_text["recall@5"],
        "image_to_text_r10": image_to_text["recall@10"],
        "image_to_text_mean_recall": image_to_text["mean_recall"],
        "image_to_text_mrr": image_to_text["mrr"],
        "image_to_text_median_rank": image_to_text["median_rank"],

        "overall_mean_recall": overall_mean_recall,
        "overall_mrr": overall_mrr,
        "evaluation_split": "validation",
        "test_set_used": False,
    }

    print("\nValidation metrics:")
    print(f"Text -> Image R@1: {metrics['text_to_image_r1']:.4f}")
    print(f"Image -> Text R@1: {metrics['image_to_text_r1']:.4f}")
    print(f"Overall Mean Recall: {overall_mean_recall:.4f}")
    print(f"Overall MRR: {overall_mrr:.4f}")

    return metrics


def get_best_training_losses(history):
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])

    if not train_losses or not val_losses:
        raise ValueError(
            "Training history does not contain train/validation losses."
        )

    if len(train_losses) != len(val_losses):
        raise ValueError(
            "Train and validation history lengths do not match."
        )

    best_epoch_index = int(np.argmin(val_losses))

    return {
        "best_epoch": best_epoch_index + 1,
        "best_train_loss": float(train_losses[best_epoch_index]),
        "best_val_loss": float(val_losses[best_epoch_index]),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "epochs_completed": len(train_losses),
    }


def build_result_row(experiment, history, metrics, duration):
    row = {
        "experiment": experiment["name"],
        "status": "completed",
        "changed_parameter": experiment["changed_parameter"],
        "changed_value": experiment["changed_value"],

        "learning_rate": experiment["learning_rate"],
        "batch_size": experiment["batch_size"],
        "output_dim": experiment["output_dim"],
        "dropout": experiment["dropout"],
        "temperature": experiment["temperature"],

        "seed": SEED,
        "fold": FOLD,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "none",
        "epochs_requested": 5,
        "early_stopping_patience": 3,
        "training_time_minutes": duration,

        **get_best_training_losses(history),
        **metrics,
    }

    return row


def save_results(rows):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = OUTPUT_DIR / "hyperparameter_results.json"
    csv_path = OUTPUT_DIR / "hyperparameter_comparison.csv"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rows,
            file,
            indent=4,
            ensure_ascii=False,
        )

    columns = [
        "experiment",
        "status",
        "changed_parameter",
        "changed_value",
        "learning_rate",
        "batch_size",
        "output_dim",
        "dropout",
        "temperature",
        "seed",
        "fold",
        "optimizer",
        "weight_decay",
        "scheduler",
        "epochs_requested",
        "epochs_completed",
        "best_epoch",
        "best_train_loss",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "training_time_minutes",
        "text_to_image_r1",
        "text_to_image_r5",
        "text_to_image_r10",
        "text_to_image_mean_recall",
        "text_to_image_mrr",
        "text_to_image_median_rank",
        "image_to_text_r1",
        "image_to_text_r5",
        "image_to_text_r10",
        "image_to_text_mean_recall",
        "image_to_text_mrr",
        "image_to_text_median_rank",
        "overall_mean_recall",
        "overall_mrr",
        "evaluation_split",
        "test_set_used",
        "checkpoint_path",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved:")
    print(csv_path)
    print(json_path)


def save_best_configuration(rows):
    successful = [
        row
        for row in rows
        if row.get("status") == "completed"
        and "overall_mean_recall" in row
    ]

    if not successful:
        print(
            "\n[WARNING] No completed experiment is available "
            "for best-configuration selection."
        )
        return None

    # Primary criterion: highest validation mean recall.
    # Tie-breaker: lowest best validation loss.
    best = max(
        successful,
        key=lambda row: (
            row["overall_mean_recall"],
            -row["best_val_loss"],
        ),
    )

    report = {
        "selection_split": "validation",
        "test_set_used_for_selection": False,
        "selection_criterion": (
            "Highest overall validation mean recall; "
            "lowest best validation loss used as tie-breaker."
        ),
        "best_experiment": best["experiment"],
        "best_configuration": {
            "learning_rate": best["learning_rate"],
            "batch_size": best["batch_size"],
            "output_dim": best["output_dim"],
            "dropout": best["dropout"],
            "temperature": best["temperature"],
        },
        "best_validation_metrics": {
            "overall_mean_recall": best["overall_mean_recall"],
            "overall_mrr": best["overall_mrr"],
            "best_val_loss": best["best_val_loss"],
        },
    }

    path = OUTPUT_DIR / "best_hyperparameter_configuration.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETER CONFIGURATION")
    print("=" * 70)
    print("Experiment:", best["experiment"])
    print("Configuration:", report["best_configuration"])
    print(
        "Validation Mean Recall:",
        f"{best['overall_mean_recall']:.6f}",
    )
    print(
        "Best Validation Loss:",
        f"{best['best_val_loss']:.6f}",
    )
    print("Selection used Validation only.")
    print("Test set was NOT used.")

    return report


def create_plots(rows):
    import matplotlib.pyplot as plt

    successful = [
        row
        for row in rows
        if row.get("status") == "completed"
        and "overall_mean_recall" in row
    ]

    if not successful:
        print("[WARNING] No successful experiments to plot.")
        return

    plot_dir = OUTPUT_DIR / "plots"
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_labels = {
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "output_dim": "Retrieval Embedding Dimension",
        "dropout": "Dropout",
        "temperature": "InfoNCE Temperature",
    }

    for parameter, title in parameter_labels.items():
        selected = [
            row
            for row in successful
            if row["changed_parameter"] in {parameter, "reference"}
        ]

        if not selected:
            continue

        selected.sort(
            key=lambda row: (
                row[parameter],
                row["experiment"],
            )
        )

        labels = [
            str(row[parameter])
            for row in selected
        ]

        mean_recalls = [
            row["overall_mean_recall"]
            for row in selected
        ]

        val_losses = [
            row["best_val_loss"]
            for row in selected
        ]

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(labels, mean_recalls)
        axis.set_title(f"{title} vs Validation Mean Recall")
        axis.set_xlabel(parameter.replace("_", " ").title())
        axis.set_ylabel("Overall Validation Mean Recall")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()

        figure.savefig(
            plot_dir / f"{parameter}_mean_recall.png",
            dpi=200,
        )
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(labels, val_losses)
        axis.set_title(f"{title} vs Best Validation Loss")
        axis.set_xlabel(parameter.replace("_", " ").title())
        axis.set_ylabel("Best Validation Loss")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()

        figure.savefig(
            plot_dir / f"{parameter}_validation_loss.png",
            dpi=200,
        )
        plt.close(figure)

    # Overall experiment comparison.
    labels = [
        row["experiment"]
        for row in successful
    ]
    values = [
        row["overall_mean_recall"]
        for row in successful
    ]

    figure, axis = plt.subplots(
        figsize=(max(10, len(labels) * 1.2), 6)
    )
    axis.bar(labels, values)
    axis.set_title("Hyperparameter Experiments - Validation Mean Recall")
    axis.set_ylabel("Overall Validation Mean Recall")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    figure.savefig(
        plot_dir / "all_experiments_mean_recall.png",
        dpi=200,
    )
    plt.close(figure)

    print("\nPlots saved in:", plot_dir)


def run_all(experiments, epochs=5):
    rows = []

    for index, experiment in enumerate(experiments, start=1):
        print("\n\n")
        print("#" * 75)
        print(
            f"EXPERIMENT {index}/{len(experiments)}: "
            f"{experiment['name']}"
        )
        print("#" * 75)

        start_time = time.time()

        try:
            history = run_experiment(
                experiment_name=experiment["name"],
                model_type="baseline",
                epochs=epochs,
                batch_size=experiment["batch_size"],
                seed=SEED,
                fold=FOLD,
                learning_rate=experiment["learning_rate"],
                temperature=experiment["temperature"],
                output_dim=experiment["output_dim"],
                dropout=experiment["dropout"],
                log_dir="runs/hyperparameter_experiments",
            )

            duration = (time.time() - start_time) / 60.0

            metrics = evaluate_experiment(experiment)

            row = build_result_row(
                experiment=experiment,
                history=history,
                metrics=metrics,
                duration=duration,
            )

            rows.append(row)

            print(
                f"\n[PASS] {experiment['name']} "
                "training and validation evaluation completed."
            )

        except tf.errors.ResourceExhaustedError as error:
            duration = (time.time() - start_time) / 60.0

            rows.append({
                "experiment": experiment["name"],
                "status": "failed_out_of_memory",
                "changed_parameter": experiment["changed_parameter"],
                "changed_value": experiment["changed_value"],
                "learning_rate": experiment["learning_rate"],
                "batch_size": experiment["batch_size"],
                "output_dim": experiment["output_dim"],
                "dropout": experiment["dropout"],
                "temperature": experiment["temperature"],
                "training_time_minutes": duration,
                "error": str(error)[:1000],
            })

            print(
                f"\n[OUT OF MEMORY] {experiment['name']} "
                "could not run with the available memory."
            )

        except Exception as error:
            duration = (time.time() - start_time) / 60.0

            rows.append({
                "experiment": experiment["name"],
                "status": "failed",
                "changed_parameter": experiment["changed_parameter"],
                "changed_value": experiment["changed_value"],
                "learning_rate": experiment["learning_rate"],
                "batch_size": experiment["batch_size"],
                "output_dim": experiment["output_dim"],
                "dropout": experiment["dropout"],
                "temperature": experiment["temperature"],
                "training_time_minutes": duration,
                "error": str(error)[:1000],
            })

            print(f"\n[FAILED] {experiment['name']}")
            print(error)
            traceback.print_exc()

        finally:
            tf.keras.backend.clear_session()
            gc.collect()

            # Save progress after every experiment.
            save_results(rows)

    save_best_configuration(rows)
    create_plots(rows)
    save_results(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run one-factor-at-a-time hyperparameter experiments "
            "for the VisionMatch AI Baseline model."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Maximum training epochs per experiment.",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Optional experiment names to run. "
            "By default, all experiments are executed."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiment configurations without training.",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive.")

    experiments = build_experiments()
    validate_experiment_design(experiments)

    if args.only:
        requested = set(args.only)
        available = {
            experiment["name"]
            for experiment in experiments
        }

        unknown = requested - available

        if unknown:
            parser.error(
                f"Unknown experiment names: {sorted(unknown)}"
            )

        experiments = [
            experiment
            for experiment in experiments
            if experiment["name"] in requested
        ]

    print("\nEXPERIMENT PLAN")
    print("=" * 90)

    for experiment in experiments:
        print(
            f"{experiment['name']:<25} "
            f"LR={experiment['learning_rate']:<8} "
            f"Batch={experiment['batch_size']:<3} "
            f"Dim={experiment['output_dim']:<3} "
            f"Dropout={experiment['dropout']:<4} "
            f"Temp={experiment['temperature']}"
        )

    if args.list:
        print("\nList-only mode. No training was started.")
        return

    print("\nWARNING:")
    print(
        "This command starts TensorFlow training for each "
        "selected experiment."
    )
    print(
        "Run it only when the GPU is available."
    )

    run_all(
        experiments=experiments,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
