
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "runs" / "hyperparameter_experiments"

RESULTS_FILE = (
    ROOT
    / "results"
    / "hyperparameter_experiments"
    / "hyperparameter_comparison.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "hyperparameter_experiments"
    / "plots"
)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {path}")


def discover_training_logs():
    if not LOG_DIR.exists():
        print(f"[WARNING] Training log directory not found: {LOG_DIR}")
        return {}

    logs = {}

    for path in sorted(LOG_DIR.glob("*.csv")):
        if path.name == "hyperparameter_comparison.csv":
            continue

        rows = read_csv(path)

        if not rows:
            continue

        required = {"epoch", "train_loss", "val_loss"}
        if not required.issubset(rows[0].keys()):
            continue

        experiment_name = path.name.rsplit("_", 2)[0]

        logs.setdefault(experiment_name, []).append({
            "path": path,
            "rows": rows,
        })

    return logs


def plot_individual_training_curves(logs):
    for experiment_name, files in logs.items():
        for file_index, log in enumerate(files, start=1):
            rows = log["rows"]

            epochs = []
            train_losses = []
            val_losses = []

            for row in rows:
                epoch = safe_float(row.get("epoch"))
                train_loss = safe_float(row.get("train_loss"))
                val_loss = safe_float(row.get("val_loss"))

                if epoch is None:
                    continue

                if train_loss is not None:
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

            if not epochs:
                continue

            fig, axis = plt.subplots(figsize=(8, 5))

            axis.plot(
                epochs,
                train_losses,
                marker="o",
                label="Training Loss",
            )

            valid_val = [
                (epoch, loss)
                for epoch, loss in zip(epochs, val_losses)
                if loss is not None
            ]

            if valid_val:
                axis.plot(
                    [item[0] for item in valid_val],
                    [item[1] for item in valid_val],
                    marker="s",
                    label="Validation Loss",
                )

            axis.set_title(
                f"Training and Validation Loss - {experiment_name}"
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.set_xticks(epochs)
            axis.grid(True, alpha=0.3)
            axis.legend()

            suffix = f"_{file_index}" if len(files) > 1 else ""

            save_figure(
                fig,
                f"{experiment_name}_loss_curve{suffix}.png",
            )


def plot_all_training_curves(logs):
    fig, axis = plt.subplots(figsize=(11, 6))
    plotted = False

    for experiment_name, files in logs.items():
        for log in files:
            rows = log["rows"]

            epochs = []
            train_losses = []

            for row in rows:
                epoch = safe_float(row.get("epoch"))
                loss = safe_float(row.get("train_loss"))

                if epoch is not None and loss is not None:
                    epochs.append(epoch)
                    train_losses.append(loss)

            if epochs:
                axis.plot(
                    epochs,
                    train_losses,
                    marker="o",
                    label=experiment_name,
                )
                plotted = True

    if not plotted:
        plt.close(fig)
        print("[INFO] No training curves available yet.")
        return

    axis.set_title("Training Loss Comparison Across Experiments")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training Loss")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    save_figure(fig, "all_experiments_training_loss.png")


def plot_all_validation_curves(logs):
    fig, axis = plt.subplots(figsize=(11, 6))
    plotted = False

    for experiment_name, files in logs.items():
        for log in files:
            rows = log["rows"]

            epochs = []
            val_losses = []

            for row in rows:
                epoch = safe_float(row.get("epoch"))
                loss = safe_float(row.get("val_loss"))

                if epoch is not None and loss is not None:
                    epochs.append(epoch)
                    val_losses.append(loss)

            if epochs:
                axis.plot(
                    epochs,
                    val_losses,
                    marker="s",
                    label=experiment_name,
                )
                plotted = True

    if not plotted:
        plt.close(fig)
        print("[INFO] No validation curves available yet.")
        return

    axis.set_title("Validation Loss Comparison Across Experiments")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Loss")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)

    save_figure(fig, "all_experiments_validation_loss.png")


def load_comparison_results():
    if not RESULTS_FILE.exists():
        print(
            "[INFO] Hyperparameter comparison CSV does not exist yet. "
            "Metric plots will be generated after experiments finish."
        )
        return []

    rows = read_csv(RESULTS_FILE)

    return [
        row
        for row in rows
        if row.get("status") == "completed"
    ]


def plot_parameter_metric(
    rows,
    parameter,
    metric,
    title,
    ylabel,
    filename,
):
    selected = [
        row
        for row in rows
        if safe_float(row.get(parameter)) is not None
        and safe_float(row.get(metric)) is not None
    ]

    if not selected:
        print(f"[INFO] No data for {filename}")
        return

    selected.sort(
        key=lambda row: safe_float(row[parameter])
    )

    labels = [
        str(row[parameter])
        for row in selected
    ]

    values = [
        safe_float(row[metric])
        for row in selected
    ]

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_xlabel(parameter.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.3)

    save_figure(fig, filename)


def create_metric_plots(rows):
    parameter_names = {
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "output_dim": "Retrieval Embedding Dimension",
        "dropout": "Dropout",
        "temperature": "InfoNCE Temperature",
    }

    metrics = [
        (
            "overall_mean_recall",
            "Overall Validation Mean Recall",
            "Overall Mean Recall",
        ),
        (
            "overall_mrr",
            "Overall Validation MRR",
            "Overall MRR",
        ),
        (
            "best_val_loss",
            "Best Validation Loss",
            "Best Validation Loss",
        ),
    ]

    for parameter, parameter_title in parameter_names.items():
        for metric, metric_title, ylabel in metrics:
            plot_parameter_metric(
                rows=rows,
                parameter=parameter,
                metric=metric,
                title=f"{parameter_title} vs {metric_title}",
                ylabel=ylabel,
                filename=f"{parameter}_{metric}.png",
            )

    for metric, title, ylabel in metrics[:2]:
        plot_parameter_metric(
            rows=rows,
            parameter="experiment",
            metric=metric,
            title=f"All Experiments - {title}",
            ylabel=ylabel,
            filename=f"all_experiments_{metric}.png",
        )


def create_summary_table(rows):
    if not rows:
        print("[INFO] No completed experiments for summary table.")
        return

    summary_fields = [
        "experiment",
        "learning_rate",
        "batch_size",
        "output_dim",
        "dropout",
        "temperature",
        "best_epoch",
        "best_train_loss",
        "best_val_loss",
        "overall_mean_recall",
        "overall_mrr",
        "training_time_minutes",
    ]

    available_fields = [
        field
        for field in summary_fields
        if field in rows[0]
    ]

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            safe_float(row.get("overall_mean_recall"))
            if safe_float(row.get("overall_mean_recall")) is not None
            else -1
        ),
        reverse=True,
    )

    output_path = OUTPUT_DIR / "hyperparameter_summary.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=available_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"[SAVED] {output_path}")

    if sorted_rows:
        best = sorted_rows[0]
        print("\nBest completed configuration:")
        print("Experiment:", best.get("experiment"))
        print("Validation Mean Recall:", best.get("overall_mean_recall"))
        print("Validation Loss:", best.get("best_val_loss"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate VisionMatch hyperparameter plots."
    )

    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="Generate training curves without requiring retrieval results.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("VISIONMATCH AI - HYPERPARAMETER VISUALIZATION")
    print("=" * 65)

    logs = discover_training_logs()

    if logs:
        print(f"\nFound {len(logs)} experiment log groups.")

        plot_individual_training_curves(logs)
        plot_all_training_curves(logs)
        plot_all_validation_curves(logs)
    else:
        print("\nNo hyperparameter training logs found yet.")

    if args.logs_only:
        print("\nLogs-only mode completed.")
        return

    results = load_comparison_results()

    if results:
        print(f"\nFound {len(results)} completed experiments.")

        create_metric_plots(results)
        create_summary_table(results)
    else:
        print(
            "\nRetrieval comparison plots are pending "
            "completed validation evaluations."
        )

    print("\nVisualization process completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
