
import torch
from torch import nn
from torch.amp import GradScaler, autocast

from src.training.train_config import TrainingConfig
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.evaluator import Evaluator
from src.training.checkpoint_manager import CheckpointManager
from src.training.training_logger import TrainingLogger
from src.training.early_stopping import EarlyStopping


class Trainer:
    """
    Training engine for VisionMatch AI.

    Handles:
        - Training epochs
        - Forward and backward passes
        - Optimizer updates
        - Mixed precision training
        - Gradient clipping
        - Learning rate scheduling
        - Validation
        - Checkpoint saving and loading
        - Training logs
        - Early stopping
        - Resume training

    Expected model output:
        image_embeddings: [B, D]
        text_embeddings:  [B, C, D]

    Expected batch:
        image:    [B, 3, H, W]
        captions: [B, C, L]
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        optimizer=None,
        scheduler=None,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")

        if not isinstance(loss_fn, nn.Module):
            raise TypeError("loss_fn must be a torch.nn.Module.")

        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance.")

        self.model = model
        self.loss_fn = loss_fn
        self.config = config

        self.device = torch.device(config.resolve_device())

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.optimizer = (
            optimizer
            if optimizer is not None
            else build_optimizer(self.model, self.config)
        )

        self.scheduler = (
            scheduler
            if scheduler is not None
            else build_scheduler(self.optimizer, self.config)
        )

        self.use_amp = (
            config.use_amp and self.device.type == "cuda"
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.global_step = 0
        self.current_epoch = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        print("Trainer initialized successfully.")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Optimizer: {type(self.optimizer).__name__}")

        if self.scheduler is not None:
            print(f"Scheduler: {type(self.scheduler).__name__}")

    # =====================================================
    # Batch Preparation
    # =====================================================

    def _prepare_batch(self, batch):
        """
        Move images and captions to the selected device.
        """

        if not isinstance(batch, dict):
            raise TypeError(
                "Each batch must be a dictionary containing "
                "'image' and 'captions'."
            )

        if "image" not in batch or "captions" not in batch:
            raise KeyError(
                "Batch must contain 'image' and 'captions' keys."
            )

        images = batch["image"].to(
            self.device,
            non_blocking=True,
        )

        captions = batch["captions"].to(
            self.device,
            non_blocking=True,
        )

        if images.ndim != 4:
            raise ValueError(
                f"Expected images [B, 3, H, W], got {images.shape}."
            )

        if images.size(1) != 3:
            raise ValueError("Images must have 3 RGB channels.")

        if captions.ndim != 3:
            raise ValueError(
                f"Expected captions [B, C, L], got {captions.shape}."
            )

        if images.size(0) != captions.size(0):
            raise ValueError(
                "Image and caption batch sizes must match."
            )

        if images.size(0) == 0 or captions.size(1) == 0:
            raise ValueError(
                "Empty image or caption batches are invalid."
            )

        return images, captions

    # =====================================================
    # Loss Computation
    # =====================================================

    def _compute_loss(self, outputs):
        """
        Calculate the selected retrieval loss.
        """

        required_keys = {
            "image_embeddings",
            "text_embeddings",
        }

        if not isinstance(outputs, dict):
            raise TypeError(
                "Model must return a dictionary of embeddings."
            )

        missing_keys = required_keys - outputs.keys()

        if missing_keys:
            raise KeyError(
                f"Missing model outputs: {missing_keys}"
            )

        image_embeddings = outputs["image_embeddings"]
        text_embeddings = outputs["text_embeddings"]

        loss = self.loss_fn(
            image_embeddings,
            text_embeddings,
        )

        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(
                "The loss function must return a scalar tensor."
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite loss detected: {loss.item()}"
            )

        return loss

    # =====================================================
    # Training Epoch
    # =====================================================

    def train_epoch(self, train_loader):
        """
        Train the model for one complete epoch.

        Returns:
            Average training loss.
        """

        self.model.train()
        self.loss_fn.train()

        total_loss = 0.0
        total_samples = 0

        for batch_index, batch in enumerate(train_loader):

            images, captions = self._prepare_batch(batch)

            batch_size = images.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):

                outputs = self.model(images, captions)

                loss = self._compute_loss(outputs)

            self.scaler.scale(loss).backward()

            if self.config.gradient_clip_norm is not None:

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

            self.global_step += 1

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Batch [{batch_index + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        if total_samples == 0:
            raise RuntimeError(
                "The training DataLoader is empty."
            )

        average_loss = total_loss / total_samples

        self.current_epoch += 1

        self.history["train_loss"].append(average_loss)

        return average_loss

    # =====================================================
    # Learning Rate Scheduler
    # =====================================================

    def step_scheduler(self, validation_loss=None):
        """
        Update the learning rate scheduler.

        ReduceLROnPlateau requires a validation loss.
        Other schedulers are stepped once per epoch.
        """

        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):

            if validation_loss is None:
                raise ValueError(
                    "ReduceLROnPlateau requires validation_loss."
                )

            self.scheduler.step(validation_loss)

        else:
            self.scheduler.step()

    # =====================================================
    # Checkpoint State Helpers
    # =====================================================

    def _get_early_stopping_state(self, early_stopping):
        """
        Extract early stopping state for checkpoint saving.
        """

        if early_stopping is None:
            return None

        return {
            "best_loss": early_stopping.best_loss,
            "counter": early_stopping.counter,
            "early_stop": early_stopping.early_stop,
        }

    def _restore_early_stopping_state(
        self,
        early_stopping,
        state,
    ):
        """
        Restore early stopping state from loaded checkpoint data.
        """

        if early_stopping is None:
            return

        if state is None:
            print(
                "No early stopping state found in checkpoint. "
                "Using a fresh early stopping state."
            )
            return

        required_keys = {
            "best_loss",
            "counter",
            "early_stop",
        }

        if not isinstance(state, dict):
            raise TypeError(
                "Checkpoint early stopping state must be a dictionary."
            )

        missing_keys = required_keys - state.keys()

        if missing_keys:
            raise KeyError(
                f"Missing early stopping state keys: {missing_keys}"
            )

        early_stopping.best_loss = state["best_loss"]
        early_stopping.counter = state["counter"]
        early_stopping.early_stop = state["early_stop"]

        print("Early stopping state restored.")

    # =====================================================
    # Complete Training Pipeline
    # =====================================================

    def fit(
        self,
        train_loader,
        epochs=None,
        val_loader=None,
        checkpoint_manager=None,
        logger=None,
        early_stopping=None,
        resume_from=None,
    ):
        """
        Run the complete training pipeline.

        Optional components:
            - Validation
            - Checkpoint saving
            - Training logging
            - Early stopping
            - Resume from checkpoint

        Args:
            train_loader: Training DataLoader.
            epochs: Total target number of epochs.
            val_loader: Optional validation DataLoader.
            checkpoint_manager: Optional CheckpointManager.
            logger: Optional TrainingLogger.
            early_stopping: Optional EarlyStopping.
            resume_from: Optional checkpoint path.

        Returns:
            Training history dictionary.
        """

        total_epochs = (
            self.config.epochs if epochs is None else epochs
        )

        if (
            isinstance(total_epochs, bool)
            or not isinstance(total_epochs, int)
            or total_epochs <= 0
        ):
            raise ValueError(
                "epochs must be a positive integer."
            )

        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ) and val_loader is None:
            raise ValueError(
                "ReduceLROnPlateau requires a validation DataLoader."
            )

        evaluator = None

        if val_loader is not None:

            evaluator = Evaluator(
                model=self.model,
                loss_fn=self.loss_fn,
                config=self.config,
            )

            if checkpoint_manager is None:
                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=self.config.checkpoint_dir
                )

            if logger is None:
                logger = TrainingLogger(
                    log_dir=self.config.log_dir,
                    experiment_name=(
                        f"visionmatch_fold_{self.config.fold}"
                    ),
                )

            if early_stopping is None:
                early_stopping = EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    min_delta=self.config.early_stopping_min_delta,
                )

        elif any(
            component is not None
            for component in (
                checkpoint_manager,
                logger,
                early_stopping,
            )
        ):
            raise ValueError(
                "Validation is required when using checkpointing, "
                "logging, or early stopping through fit()."
            )

        start_epoch = 0

        # -------------------------------------------------
        # Resume from checkpoint
        # -------------------------------------------------

        if resume_from is not None:

            # Use a temporary manager if training is running
            # without validation. This allows loading a checkpoint
            # without enabling checkpoint saving during this run.
            resume_manager = checkpoint_manager

            if resume_manager is None:
                resume_manager = CheckpointManager(
                    checkpoint_dir=self.config.checkpoint_dir
                )

            loaded_info = resume_manager.load_checkpoint(
                checkpoint_path=resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )

            start_epoch = loaded_info["epoch"] + 1

            self.current_epoch = start_epoch

            self.global_step = loaded_info["global_step"]

            self._restore_early_stopping_state(
                early_stopping,
                loaded_info.get("early_stopping_state"),
            )

            print(
                f"Training resumed. "
                f"Next epoch: {start_epoch + 1}."
            )

        if start_epoch >= total_epochs:
            raise ValueError(
                "The checkpoint epoch has already reached "
                "or exceeded the requested total epochs."
            )

        # -------------------------------------------------
        # Training Loop
        # -------------------------------------------------

        print("\nStarting VisionMatch AI Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {total_epochs}")
        print(f"Starting Epoch: {start_epoch + 1}")
        print("-" * 50)

        for epoch in range(start_epoch, total_epochs):

            print(
                f"\nEpoch [{epoch + 1}/{total_epochs}]"
            )

            if logger is not None:
                logger.start_epoch()

            # Capture the LR actually used during this epoch.
            epoch_learning_rate = (
                self.optimizer.param_groups[0]["lr"]
            )

            # -------------------------------
            # Training
            # -------------------------------

            train_loss = self.train_epoch(train_loader)

            # -------------------------------
            # Validation
            # -------------------------------

            val_loss = None

            if evaluator is not None:

                validation_results = evaluator.evaluate(
                    val_loader
                )

                val_loss = validation_results["val_loss"]

                self.history["val_loss"].append(val_loss)

                print(f"Validation Loss: {val_loss:.4f}")

            # -------------------------------
            # History
            # -------------------------------

            self.history["learning_rate"].append(
                epoch_learning_rate
            )

            # -------------------------------
            # Early Stopping Update
            # -------------------------------

            should_stop = False

            if early_stopping is not None:

                if val_loss is None:
                    raise RuntimeError(
                        "Early stopping requires a validation loss."
                    )

                should_stop = early_stopping(val_loss)

            # -------------------------------
            # Scheduler Update
            # -------------------------------

            if self.scheduler is not None:

                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.step_scheduler(
                        validation_loss=val_loss
                    )
                else:
                    self.step_scheduler()

            next_lr = self.optimizer.param_groups[0]["lr"]

            # -------------------------------
            # Checkpointing
            # -------------------------------

            if checkpoint_manager is not None:

                if val_loss is None:
                    raise RuntimeError(
                        "Checkpoint saving requires validation loss."
                    )

                is_best = checkpoint_manager.update_best(
                    val_loss
                )

                checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                    global_step=self.global_step,
                    config=self.config,
                    is_best=is_best,
                    early_stopping_state=(
                        self._get_early_stopping_state(
                            early_stopping
                        )
                    ),
                    save_best_only=self.config.save_best_only,
                )

            # -------------------------------
            # Training Logger
            # -------------------------------

            if logger is not None:

                logger.log_epoch(
                    epoch=self.current_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=epoch_learning_rate,
                )

            # -------------------------------
            # Epoch Summary
            # -------------------------------

            print(f"Training Loss: {train_loss:.4f}")
            print(
                f"Learning Rate Used: "
                f"{epoch_learning_rate:.8f}"
            )
            print(
                f"Next Learning Rate: {next_lr:.8f}"
            )

            # -------------------------------
            # Early Stopping Decision
            # -------------------------------

            if should_stop:

                print(
                    "\nEarly stopping: training terminated."
                )

                break

        print("\nTraining finished.")

        return self.history


if __name__ == "__main__":

    print("Trainer module imported successfully.")
    print("Use Trainer(model, loss_fn, config) to start training.")
