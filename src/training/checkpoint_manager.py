import json
import math
from pathlib import Path

import tensorflow as tf


class CheckpointManager:
    """Save and restore TensorFlow model and optimizer checkpoints."""

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.latest_dir = self.checkpoint_dir / "latest"
        self.best_dir = self.checkpoint_dir / "best"

        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")

        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.val_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)
        self.best_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)

        self.checkpoint = None
        self.latest_manager = None
        self.best_manager = None

    @staticmethod
    def _validate_finite_number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

    @staticmethod
    def _validate_early_stopping_state(state):
        if state is None:
            return

        if not isinstance(state, dict):
            raise TypeError("early_stopping_state must be a dictionary or None.")

        required = {"best_loss", "counter", "early_stop"}
        missing = required - state.keys()

        if missing:
            raise KeyError(f"Missing early stopping keys: {missing}")

        if not isinstance(state["counter"], int) or state["counter"] < 0:
            raise ValueError("Early stopping counter must be non-negative.")

        if not isinstance(state["early_stop"], bool):
            raise TypeError("early_stop must be boolean.")

        best_loss = state["best_loss"]
        if not isinstance(best_loss, (int, float)) or math.isnan(float(best_loss)):
            raise ValueError("Invalid early stopping best_loss.")

    def _build_managers(self, model, optimizer=None):
        objects = {
            "model": model,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "val_loss": self.val_loss,
            "best_loss": self.best_loss,
        }

        if optimizer is not None:
            objects["optimizer"] = optimizer

        self.checkpoint = tf.train.Checkpoint(**objects)

        self.latest_manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.latest_dir),
            max_to_keep=3,
            checkpoint_name="latest_checkpoint",
        )

        self.best_manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.best_dir),
            max_to_keep=1,
            checkpoint_name="best_model",
        )

    def update_best(self, val_loss):
        self._validate_finite_number(val_loss, "val_loss")

        if float(val_loss) < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.best_loss.assign(self.best_val_loss)
            return True

        return False

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
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")

        if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer.")

        self._validate_finite_number(val_loss, "val_loss")
        self._validate_early_stopping_state(early_stopping_state)

        if self.checkpoint is None:
            self._build_managers(model, optimizer)

        self.epoch.assign(epoch)
        self.global_step.assign(global_step)
        self.val_loss.assign(float(val_loss))

        if is_best:
            self.best_val_loss = float(val_loss)
            self.best_loss.assign(self.best_val_loss)

        latest_path = self.latest_manager.save(
            checkpoint_number=epoch + 1
        )

        config_data = config.to_dict() if hasattr(config, "to_dict") else config

        metadata = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "best_val_loss": float(self.best_val_loss),
            "global_step": global_step,
            "config": config_data,
            "early_stopping_state": early_stopping_state,
        }

        metadata_path = latest_path + ".json"
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, default=str)

        print(f"Latest checkpoint saved: {latest_path}")

        if is_best:
            best_path = self.best_manager.save(
                checkpoint_number=epoch + 1
            )

            with open(best_path + ".json", "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4, default=str)

            print(f"Best checkpoint saved: {best_path}")

        if not save_best_only:
            epoch_dir = self.checkpoint_dir / f"epoch_{epoch + 1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)

            epoch_checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=self.epoch,
                global_step=self.global_step,
                val_loss=self.val_loss,
                best_loss=self.best_loss,
            )

            epoch_checkpoint.write(str(epoch_dir / "checkpoint"))

        return latest_path

    def load_checkpoint(
        self,
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location=None,
    ):
        path = Path(checkpoint_path)

        if path.is_dir():
            latest = tf.train.latest_checkpoint(str(path))
            if latest is None:
                raise FileNotFoundError(f"No TensorFlow checkpoint found in: {path}")
            checkpoint_prefix = latest
        else:
            checkpoint_prefix = str(path)

            if not (
                Path(checkpoint_prefix + ".index").is_file()
                or Path(checkpoint_prefix).is_file()
            ):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_prefix}"
                )

        if self.checkpoint is None:
            self._build_managers(model, optimizer)

        status = self.checkpoint.restore(checkpoint_prefix)
        status.expect_partial()

        metadata_path = checkpoint_prefix + ".json"

        if Path(metadata_path).is_file():
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        else:
            metadata = {
                "epoch": int(self.epoch.numpy()),
                "val_loss": float(self.val_loss.numpy()),
                "best_val_loss": float(self.best_loss.numpy()),
                "global_step": int(self.global_step.numpy()),
                "config": None,
                "early_stopping_state": None,
            }

        self.best_val_loss = float(metadata["best_val_loss"])

        print(f"Checkpoint restored: {checkpoint_prefix}")
        print(f"Resuming from epoch: {metadata['epoch'] + 1}")

        return metadata


if __name__ == "__main__":
    print("TensorFlow CheckpointManager module imported successfully.")