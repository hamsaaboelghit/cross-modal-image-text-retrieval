
from dataclasses import dataclass, asdict
from typing import Optional
import math


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

    # Cross-modal matching loss
    matching_loss_weight: float = 0.2

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

        # Training settings validation
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer.")

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")

        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be a finite positive number."
            )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError(
                "weight_decay must be finite and non-negative."
            )

        # Optimizer validation
        if self.optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(
                "optimizer must be one of: adam, adamw, sgd."
            )

        # Scheduler validation
        if self.scheduler not in {
            "none", "step", "cosine", "plateau"
        }:
            raise ValueError(
                "scheduler must be one of: none, step, cosine, plateau."
            )

        if (
            isinstance(self.scheduler_step_size, bool)
            or not isinstance(self.scheduler_step_size, int)
            or self.scheduler_step_size <= 0
        ):
            raise ValueError(
                "scheduler_step_size must be a positive integer."
            )

        if (
            not math.isfinite(self.scheduler_gamma)
            or not 0 < self.scheduler_gamma <= 1
        ):
            raise ValueError(
                "scheduler_gamma must be finite and between 0 and 1."
            )

        if (
            isinstance(self.scheduler_patience, bool)
            or not isinstance(self.scheduler_patience, int)
            or self.scheduler_patience < 0
        ):
            raise ValueError(
                "scheduler_patience must be a non-negative integer."
            )

        if (
            not math.isfinite(self.min_learning_rate)
            or self.min_learning_rate < 0
        ):
            raise ValueError(
                "min_learning_rate must be finite and non-negative."
            )

        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate cannot exceed learning_rate."
            )

        # Retrieval loss validation
        if self.loss_type not in {
            "info_nce", "triplet", "margin"
        }:
            raise ValueError(
                "loss_type must be one of: info_nce, triplet, margin."
            )

        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError(
                "temperature must be a finite positive number."
            )

        if (
            not math.isfinite(self.margin)
            or not 0 < self.margin <= 2
        ):
            raise ValueError(
                "margin must be finite, greater than 0, and at most 2."
            )

        # Cross-modal matching loss validation
        if (
            not math.isfinite(self.matching_loss_weight)
            or self.matching_loss_weight < 0
        ):
            raise ValueError(
                "matching_loss_weight must be finite and non-negative."
            )

        # Training stability validation
        if self.gradient_clip_norm is not None:
            if (
                not math.isfinite(self.gradient_clip_norm)
                or self.gradient_clip_norm <= 0
            ):
                raise ValueError(
                    "gradient_clip_norm must be finite and positive or None."
                )

        if not isinstance(self.use_amp, bool):
            raise TypeError("use_amp must be a boolean.")

        # Data pipeline validation
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError(
                "num_workers must be a non-negative integer."
            )

        for field_name in ("pin_memory", "drop_last", "save_best_only"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean.")

        # Reproducibility validation
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")

        if (
            isinstance(self.fold, bool)
            or not isinstance(self.fold, int)
            or self.fold not in range(1, 6)
        ):
            raise ValueError("fold must be an integer between 1 and 5.")

        # Device validation
        if self.device not in {"auto", "cpu"}:
            raise ValueError(
                "device must be one of: auto, cpu."
            )

        # Directory validation
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ValueError(
                "checkpoint_dir must be a non-empty string."
            )

        if not isinstance(self.log_dir, str) or not self.log_dir.strip():
            raise ValueError(
                "log_dir must be a non-empty string."
            )

        # Early stopping validation
        if (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 0
        ):
            raise ValueError(
                "early_stopping_patience must be a non-negative integer."
            )

        if (
            not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError(
                "early_stopping_min_delta must be finite and non-negative."
            )

    def resolve_device(self) -> str:
        """Resolve the TensorFlow execution device."""

        if self.device == "cpu":
            return "/CPU:0"

        return (
            "/GPU:0"
            if self._gpu_available()
            else "/CPU:0"
        )

    @staticmethod
    def _gpu_available() -> bool:
        import tensorflow as tf

        return bool(
            tf.config.list_physical_devices("GPU")
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def display(self) -> None:
        print("\n" + "=" * 50)
        print("VISIONMATCH AI - TRAINING CONFIGURATION")
        print("=" * 50)

        for key, value in self.to_dict().items():
            print(f"{key:30}: {value}")

        print(
            f"{'resolved_device':30}: {self.resolve_device()}"
        )

        print("=" * 50)


def _run_config_tests():
    print("TrainingConfig Tests")
    print("-" * 50)

    config = TrainingConfig()
    assert config.matching_loss_weight == 0.2
    assert config.epochs == 30
    assert config.batch_size == 32

    print("Default configuration: PASS")
    print("Matching loss weight: PASS")

    # Valid custom matching weight
    config = TrainingConfig(
        matching_loss_weight=0.5
    )
    assert config.matching_loss_weight == 0.5

    print("Custom matching weight: PASS")

    # Zero matching weight disables its contribution.
    config = TrainingConfig(
        matching_loss_weight=0.0
    )
    assert config.matching_loss_weight == 0.0

    print("Zero matching weight: PASS")

    # Invalid matching weight
    invalid_values = [-0.1, float("nan"), float("inf")]

    for value in invalid_values:
        try:
            TrainingConfig(matching_loss_weight=value)
            raise AssertionError(
                f"Invalid matching_loss_weight accepted: {value}"
            )
        except ValueError:
            pass

    print("Invalid matching weight validation: PASS")

    # Invalid minimum learning rate
    try:
        TrainingConfig(
            learning_rate=1e-4,
            min_learning_rate=1e-3,
        )
        raise AssertionError(
            "min_learning_rate greater than learning_rate was accepted."
        )
    except ValueError:
        pass

    print("Learning rate validation: PASS")

    # Serialization
    config = TrainingConfig()
    config_dict = config.to_dict()

    assert "matching_loss_weight" in config_dict
    assert config_dict["matching_loss_weight"] == 0.2

    print("Configuration serialization: PASS")

    print("-" * 50)
    print("All TrainingConfig tests passed!")


if __name__ == "__main__":
    _run_config_tests()
