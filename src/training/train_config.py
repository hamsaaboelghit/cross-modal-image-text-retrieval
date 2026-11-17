from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Central configuration for training and validation.
    TensorFlow / Keras version.
    """

    # Training settings
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"

    # Scheduler
    scheduler: str = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 3
    min_learning_rate: float = 1e-7

    # Loss
    loss_type: str = "info_nce"
    temperature: float = 0.07
    margin: float = 0.2

    # Training stability
    gradient_clip_norm: Optional[float] = 1.0
    use_amp: bool = False

    # Data pipeline
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False

    # Reproducibility
    seed: int = 42
    fold: int = 1

    # Device
    device: str = "auto"

    # Saving and logging
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    save_best_only: bool = True

    # Early stopping
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(
                "optimizer must be one of: adam, adamw, sgd."
            )

        if self.scheduler not in {
            "none", "step", "cosine", "plateau"
        }:
            raise ValueError(
                "scheduler must be one of: none, step, cosine, plateau."
            )

        if self.scheduler_step_size <= 0:
            raise ValueError(
                "scheduler_step_size must be greater than 0."
            )

        if not 0 < self.scheduler_gamma <= 1:
            raise ValueError(
                "scheduler_gamma must be between 0 and 1."
            )

        if self.scheduler_patience < 0:
            raise ValueError("scheduler_patience cannot be negative.")

        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative.")

        if self.loss_type not in {
            "info_nce", "triplet", "margin"
        }:
            raise ValueError(
                "loss_type must be one of: info_nce, triplet, margin."
            )

        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        if not 0 < self.margin <= 2:
            raise ValueError("margin must be greater than 0 and at most 2.")

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive or None."
            )

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.seed < 0:
            raise ValueError("seed cannot be negative.")

        if self.fold not in range(1, 6):
            raise ValueError("fold must be between 1 and 5.")

        if self.device not in {"auto", "cpu"}:
            raise ValueError(
                "device must be one of: auto, cpu."
            )

        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience cannot be negative."
            )

        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "early_stopping_min_delta cannot be negative."
            )

    def resolve_device(self) -> str:
        """Resolve the TensorFlow execution device."""

        if self.device == "cpu":
            return "/CPU:0"

        return "/GPU:0" if self._gpu_available() else "/CPU:0"

    @staticmethod
    def _gpu_available() -> bool:
        import tensorflow as tf
        return bool(tf.config.list_physical_devices("GPU"))

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> None:
        print("\n" + "=" * 45)
        print("VISIONMATCH AI - TRAINING CONFIGURATION")
        print("=" * 45)

        for key, value in self.to_dict().items():
            print(f"{key:28}: {value}")

        print(f"{'resolved_device':28}: {self.resolve_device()}")
        print("=" * 45)


if __name__ == "__main__":
    config = TrainingConfig()
    config.display()