
import math
import torch
from pathlib import Path


class CheckpointManager:
    """
    Handles saving and loading training checkpoints.

    Saves:
        - Model state
        - Optimizer state
        - Scheduler state
        - Current epoch
        - Current validation loss
        - Best validation loss
        - Global training step
        - Training configuration
        - Early stopping state

    Always maintains:
        - latest_checkpoint.pth

    Also maintains:
        - best_model.pth when validation improves
        - Individual epoch checkpoints when save_best_only=False
    """

    def __init__(self, checkpoint_dir="checkpoints"):

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_val_loss = float("inf")

    # =====================================================
    # Validation Helpers
    # =====================================================

    @staticmethod
    def _validate_finite_number(value, name):
        """
        Validate that a value is a finite numeric value.
        """

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

    @staticmethod
    def _validate_early_stopping_state(state):
        """
        Validate the optional early stopping state.
        """

        if state is None:
            return

        if not isinstance(state, dict):
            raise TypeError(
                "early_stopping_state must be a dictionary or None."
            )

        required_keys = {
            "best_loss",
            "counter",
            "early_stop",
        }

        missing_keys = required_keys - state.keys()

        if missing_keys:
            raise KeyError(
                f"Invalid early stopping state. "
                f"Missing keys: {missing_keys}"
            )

        best_loss = state["best_loss"]
        counter = state["counter"]
        early_stop = state["early_stop"]

        if (
            isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
        ):
            raise TypeError(
                "Early stopping best_loss must be numeric."
            )

        if math.isnan(float(best_loss)):
            raise ValueError(
                "Early stopping best_loss cannot be NaN."
            )

        if (
            isinstance(counter, bool)
            or not isinstance(counter, int)
            or counter < 0
        ):
            raise ValueError(
                "Early stopping counter must be a non-negative integer."
            )

        if not isinstance(early_stop, bool):
            raise TypeError(
                "Early stopping early_stop must be a boolean."
            )

    # =====================================================
    # Save Checkpoint
    # =====================================================

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        val_loss,
        global_step=0,
        config=None,
        is_best=False,
        early_stopping_state=None,
        save_best_only=True,
    ):
        """
        Save a complete training checkpoint.

        Args:
            model: Model being trained.
            optimizer: Optimizer instance or None.
            scheduler: Scheduler instance or None.
            epoch: Current zero-based epoch index.
            val_loss: Current validation loss.
            global_step: Total optimizer update steps.
            config: Training configuration.
            is_best: Whether validation improved.
            early_stopping_state: Optional early stopping state.
            save_best_only: If False, save an additional
                checkpoint for every epoch.

        Returns:
            Path to the latest checkpoint.
        """

        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError(
                "epoch must be a non-negative integer."
            )

        self._validate_finite_number(
            val_loss,
            "val_loss",
        )

        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError(
                "global_step must be a non-negative integer."
            )

        if not isinstance(is_best, bool):
            raise TypeError(
                "is_best must be a boolean."
            )

        if not isinstance(save_best_only, bool):
            raise TypeError(
                "save_best_only must be a boolean."
            )

        self._validate_early_stopping_state(
            early_stopping_state
        )

        # Update best validation loss before building
        # the checkpoint dictionary.
        if is_best:
            self.best_val_loss = float(val_loss)

        # Convert configuration into a serializable dictionary.
        if config is not None and hasattr(config, "to_dict"):
            config_data = config.to_dict()
        else:
            config_data = config

        checkpoint = {
            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "val_loss": float(val_loss),

            "best_val_loss": float(self.best_val_loss),

            "global_step": global_step,

            "config": config_data,

            "early_stopping_state": early_stopping_state,
        }

        # -------------------------------------------------
        # Always save latest checkpoint.
        # This is required for reliable training resumption.
        # -------------------------------------------------

        checkpoint_path = (
            self.checkpoint_dir / "latest_checkpoint.pth"
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

        # -------------------------------------------------
        # Save best model only when validation improves.
        # -------------------------------------------------

        if is_best:

            best_path = (
                self.checkpoint_dir / "best_model.pth"
            )

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"Best model saved: {best_path}"
            )

        # -------------------------------------------------
        # Optional per-epoch checkpoints.
        # -------------------------------------------------

        if not save_best_only:

            epoch_path = (
                self.checkpoint_dir
                / f"checkpoint_epoch_{epoch + 1}.pth"
            )

            torch.save(
                checkpoint,
                epoch_path,
            )

            print(
                f"Epoch checkpoint saved: {epoch_path}"
            )

        return checkpoint_path

    # =====================================================
    # Load Checkpoint
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
    ):
        """
        Restore model and training states.

        Returns:
            Dictionary containing:
                - epoch
                - val_loss
                - best_val_loss
                - global_step
                - config
                - early_stopping_state
        """

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Invalid checkpoint format."
            )

        required_keys = {
            "epoch",
            "model_state_dict",
            "val_loss",
            "best_val_loss",
            "global_step",
        }

        missing_keys = required_keys - checkpoint.keys()

        if missing_keys:
            raise KeyError(
                f"Invalid checkpoint. Missing keys: {missing_keys}"
            )

        # Validate checkpoint metadata before restoring.
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
        ):
            raise ValueError(
                "Invalid checkpoint epoch."
            )

        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError(
                "Invalid checkpoint global_step."
            )

        self._validate_finite_number(
            checkpoint["val_loss"],
            "checkpoint val_loss",
        )

        best_val_loss = checkpoint["best_val_loss"]

        if (
            isinstance(best_val_loss, bool)
            or not isinstance(best_val_loss, (int, float))
        ):
            raise TypeError(
                "Checkpoint best_val_loss must be numeric."
            )

        if math.isnan(float(best_val_loss)):
            raise ValueError(
                "Checkpoint best_val_loss cannot be NaN."
            )

        early_stopping_state = checkpoint.get(
            "early_stopping_state"
        )

        self._validate_early_stopping_state(
            early_stopping_state
        )

        # Restore model.
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Restore optimizer.
        if (
            optimizer is not None
            and checkpoint.get("optimizer_state_dict") is not None
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        # Restore scheduler.
        if (
            scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.best_val_loss = float(best_val_loss)

        print(
            f"Checkpoint loaded: {checkpoint_path}"
        )

        print(
            f"Resuming from epoch: {epoch + 1}"
        )

        print(
            f"Best validation loss: {self.best_val_loss:.4f}"
        )

        return {
            "epoch": epoch,
            "val_loss": checkpoint["val_loss"],
            "best_val_loss": self.best_val_loss,
            "global_step": global_step,
            "config": checkpoint.get("config"),
            "early_stopping_state": early_stopping_state,
        }

    # =====================================================
    # Best Validation Loss
    # =====================================================

    def update_best(self, val_loss):
        """
        Check whether the current validation loss is the best.
        """

        self._validate_finite_number(
            val_loss,
            "val_loss",
        )

        if val_loss < self.best_val_loss:

            self.best_val_loss = float(val_loss)

            return True

        return False


if __name__ == "__main__":

    print(
        "CheckpointManager module imported successfully."
    )

    print(
        "Use CheckpointManager() to save and load checkpoints."
    )
