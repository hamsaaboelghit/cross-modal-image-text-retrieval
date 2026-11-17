
import json
import math
from pathlib import Path

import tensorflow as tf


class CheckpointManager:
    """Manage TensorFlow latest and best training checkpoints."""

    SCHEDULER_STATE_FIELDS = (
        "best",
        "wait",
        "cooldown_counter",
        "last_epoch",
    )

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latest_dir = self.checkpoint_dir / "latest"
        self.best_dir = self.checkpoint_dir / "best"

        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")

        self.epoch = tf.Variable(
            0, dtype=tf.int64, trainable=False
        )
        self.global_step = tf.Variable(
            0, dtype=tf.int64, trainable=False
        )
        self.val_loss = tf.Variable(
            float("inf"), dtype=tf.float32, trainable=False
        )
        self.best_loss = tf.Variable(
            float("inf"), dtype=tf.float32, trainable=False
        )

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
            raise TypeError(
                "early_stopping_state must be a dictionary or None."
            )

        required = {"best_loss", "counter", "early_stop"}
        missing = required - state.keys()

        if missing:
            raise KeyError(
                f"Missing early stopping keys: {missing}"
            )

        if (
            not isinstance(state["counter"], int)
            or isinstance(state["counter"], bool)
            or state["counter"] < 0
        ):
            raise ValueError(
                "Early stopping counter must be non-negative."
            )

        if not isinstance(state["early_stop"], bool):
            raise TypeError("early_stop must be boolean.")

        best_loss = state["best_loss"]

        if (
            not isinstance(best_loss, (int, float))
            or math.isnan(float(best_loss))
        ):
            raise ValueError(
                "Invalid early stopping best_loss."
            )

    @classmethod
    def _get_scheduler_state(cls, scheduler):
        if scheduler is None:
            return None

        state = {}

        for field in cls.SCHEDULER_STATE_FIELDS:
            if not hasattr(scheduler, field):
                continue

            value = getattr(scheduler, field)

            if isinstance(value, (int, float, bool, str)):
                state[field] = value

        return state

    @classmethod
    def _restore_scheduler_state(cls, scheduler, state):
        if scheduler is None or not state:
            return

        for field in cls.SCHEDULER_STATE_FIELDS:
            if field in state and hasattr(scheduler, field):
                setattr(scheduler, field, state[field])

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

    @staticmethod
    def _write_metadata(checkpoint_path, metadata):
        metadata_path = checkpoint_path + ".json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
                default=str,
            )

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
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 0
        ):
            raise ValueError(
                "epoch must be a non-negative integer."
            )

        if (
            not isinstance(global_step, int)
            or isinstance(global_step, bool)
            or global_step < 0
        ):
            raise ValueError(
                "global_step must be a non-negative integer."
            )

        self._validate_finite_number(val_loss, "val_loss")
        self._validate_early_stopping_state(
            early_stopping_state
        )

        if self.checkpoint is None:
            self._build_managers(model, optimizer)

        self.epoch.assign(epoch)
        self.global_step.assign(global_step)
        self.val_loss.assign(float(val_loss))

        if is_best:
            self.best_val_loss = float(val_loss)
            self.best_loss.assign(self.best_val_loss)

        config_data = (
            config.to_dict()
            if hasattr(config, "to_dict")
            else config
        )

        metadata = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "best_val_loss": float(self.best_val_loss),
            "global_step": global_step,
            "config": config_data,
            "early_stopping_state": early_stopping_state,
            "scheduler_state": self._get_scheduler_state(
                scheduler
            ),
        }

        latest_path = self.latest_manager.save(
            checkpoint_number=epoch + 1
        )

        self._write_metadata(latest_path, metadata)

        print(f"Latest checkpoint saved: {latest_path}")

        if is_best:
            best_path = self.best_manager.save(
                checkpoint_number=epoch + 1
            )

            self._write_metadata(best_path, metadata)

            print(f"Best checkpoint saved: {best_path}")

        if not save_best_only:
            epoch_dir = (
                self.checkpoint_dir / f"epoch_{epoch + 1}"
            )
            epoch_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            epoch_checkpoint = tf.train.Checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=self.epoch,
                global_step=self.global_step,
                val_loss=self.val_loss,
                best_loss=self.best_loss,
            )

            epoch_checkpoint.write(
                str(epoch_dir / "checkpoint")
            )

        return latest_path

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path):
        path = Path(checkpoint_path)

        if path.is_dir():
            # If the experiment root is supplied, resume from latest/.
            latest_dir = path / "latest"

            if latest_dir.is_dir():
                latest = tf.train.latest_checkpoint(
                    str(latest_dir)
                )

                if latest is not None:
                    return latest

            # If latest/ was supplied directly, use it.
            latest = tf.train.latest_checkpoint(str(path))

            if latest is not None:
                return latest

            raise FileNotFoundError(
                f"No TensorFlow checkpoint found in: {path}"
            )

        checkpoint_prefix = str(path)

        if not Path(checkpoint_prefix + ".index").is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_prefix}"
            )

        return checkpoint_prefix

    def load_checkpoint(
        self,
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location=None,
    ):
        checkpoint_prefix = self._resolve_checkpoint_path(
            checkpoint_path
        )

        if self.checkpoint is None:
            self._build_managers(model, optimizer)

        status = self.checkpoint.restore(checkpoint_prefix)

        # Optimizer slots may be restored lazily by TensorFlow.
        status.expect_partial()

        metadata_path = checkpoint_prefix + ".json"

        if Path(metadata_path).is_file():
            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)
        else:
            metadata = {
                "epoch": int(self.epoch.numpy()),
                "val_loss": float(self.val_loss.numpy()),
                "best_val_loss": float(self.best_loss.numpy()),
                "global_step": int(self.global_step.numpy()),
                "config": None,
                "early_stopping_state": None,
                "scheduler_state": None,
            }

        required = {
            "epoch",
            "val_loss",
            "best_val_loss",
            "global_step",
        }

        missing = required - metadata.keys()

        if missing:
            raise ValueError(
                f"Checkpoint metadata is missing: {missing}"
            )

        self.best_val_loss = float(
            metadata["best_val_loss"]
        )

        self._restore_scheduler_state(
            scheduler,
            metadata.get("scheduler_state"),
        )

        print(
            f"Checkpoint restored: {checkpoint_prefix}"
        )
        print(
            f"Resuming from epoch: {int(metadata['epoch']) + 1}"
        )
        print(
            f"Restored global step: {metadata['global_step']}"
        )

        return metadata


if __name__ == "__main__":
    print(
        "TensorFlow CheckpointManager module imported successfully."
    )
