import csv
import time
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """Write comparable retrieval and auxiliary-loss metrics to CSV."""

    FIELDNAMES = [
        "epoch",
        "train_loss",
        "train_retrieval_loss",
        "train_matching_loss",
        "val_loss",
        "val_retrieval_loss",
        "val_matching_loss",
        "val_total_loss",
        "learning_rate",
        "epoch_duration",
        "timestamp",
    ]

    def __init__(self, log_dir="runs", experiment_name="visionmatch"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.csv"
        self.start_time = None

        with open(self.log_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

        print(f"Training log created: {self.log_file}")

    def start_epoch(self):
        self.start_time = time.perf_counter()

    @staticmethod
    def _optional_non_negative(value, name):
        if value is None:
            return ""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric or None.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
        return float(value)

    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss,
        learning_rate,
        train_retrieval_loss=None,
        train_matching_loss=None,
        val_retrieval_loss=None,
        val_matching_loss=None,
        val_total_loss=None,
    ):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch must be a positive integer.")
        if train_loss is None or train_loss < 0:
            raise ValueError("train_loss must be non-negative.")
        if val_loss is not None and val_loss < 0:
            raise ValueError("val_loss must be non-negative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.start_time is None:
            raise RuntimeError("Call start_epoch() before log_epoch().")

        duration = time.perf_counter() - self.start_time
        retrieval_value = (
            val_retrieval_loss if val_retrieval_loss is not None else val_loss
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_retrieval_loss": self._optional_non_negative(
                train_retrieval_loss, "train_retrieval_loss"
            ),
            "train_matching_loss": self._optional_non_negative(
                train_matching_loss, "train_matching_loss"
            ),
            "val_loss": (
                float(retrieval_value) if retrieval_value is not None else ""
            ),
            "val_retrieval_loss": self._optional_non_negative(
                retrieval_value, "val_retrieval_loss"
            ),
            "val_matching_loss": self._optional_non_negative(
                val_matching_loss, "val_matching_loss"
            ),
            "val_total_loss": self._optional_non_negative(
                val_total_loss, "val_total_loss"
            ),
            "learning_rate": float(learning_rate),
            "epoch_duration": round(duration, 3),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        with open(self.log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

        self.start_time = None
        print(
            f"Epoch {epoch} logged | Train objective: {train_loss:.4f} | "
            f"Validation retrieval loss: "
            f"{retrieval_value if retrieval_value is not None else 'N/A'}"
        )

    def get_log_path(self):
        return self.log_file

    def display_summary(self):
        print("\n" + "=" * 45)
        print("VISIONMATCH AI - TRAINING LOG")
        print("=" * 45)
        print(f"Experiment: {self.experiment_name}")
        print(f"Log file: {self.log_file}")
        print("=" * 45)


if __name__ == "__main__":
    logger = TrainingLogger(
        log_dir="runs",
        experiment_name="test_experiment",
    )
    logger.start_epoch()
    logger.log_epoch(
        epoch=1,
        train_loss=0.85,
        val_loss=0.92,
        learning_rate=1e-4,
        train_retrieval_loss=0.82,
        val_matching_loss=0.31,
        val_total_loss=0.982,
    )
    logger.display_summary()
    print("\nTrainingLogger test: PASS")
