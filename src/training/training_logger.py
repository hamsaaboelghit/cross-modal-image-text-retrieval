
import csv
import time
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    Logs training and validation metrics for VisionMatch AI.

    Features:
        - CSV experiment logging
        - Training and validation loss
        - Learning rate tracking
        - Epoch duration
        - Experiment metadata
    """

    FIELDNAMES = [
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "epoch_duration",
        "timestamp",
    ]

    def __init__(self, log_dir="runs", experiment_name="visionmatch"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_file = (
            self.log_dir
            / f"{experiment_name}_{timestamp}.csv"
        )

        self.start_time = None

        # Create CSV file and write headers.
        with open(
            self.log_file,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )
            writer.writeheader()

        print(f"Training log created: {self.log_file}")

    def start_epoch(self):
        """Start measuring epoch duration."""

        self.start_time = time.perf_counter()

    def log_epoch(
        self,
        epoch,
        train_loss,
        val_loss,
        learning_rate,
    ):
        """
        Record metrics for one epoch.
        """

        if epoch <= 0:
            raise ValueError("epoch must be greater than 0.")

        if train_loss is None or train_loss < 0:
            raise ValueError("train_loss must be non-negative.")

        if val_loss is not None and val_loss < 0:
            raise ValueError("val_loss must be non-negative.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.start_time is None:
            raise RuntimeError(
                "Call start_epoch() before log_epoch()."
            )

        duration = time.perf_counter() - self.start_time

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": (
                float(val_loss)
                if val_loss is not None else ""
            ),
            "learning_rate": float(learning_rate),
            "epoch_duration": round(duration, 3),
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        with open(
            self.log_file,
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )
            writer.writerow(row)

        self.start_time = None

        print(
            f"Epoch {epoch} logged successfully | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: "
            f"{val_loss if val_loss is not None else 'N/A'}"
        )

    def get_log_path(self):
        """Return the CSV log file path."""

        return self.log_file

    def display_summary(self):
        """Display the location of the experiment log."""

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

    # Synthetic values for testing only.
    logger.log_epoch(
        epoch=1,
        train_loss=0.85,
        val_loss=0.92,
        learning_rate=1e-4,
    )

    logger.display_summary()

    print("\nTrainingLogger test: PASS")
