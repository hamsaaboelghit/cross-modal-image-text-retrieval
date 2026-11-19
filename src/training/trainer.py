
import tensorflow as tf

from src.training.train_config import TrainingConfig
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.evaluator import Evaluator
from src.training.checkpoint_manager import CheckpointManager
from src.training.matching_loss import MatchingBinaryCrossEntropyLoss


class Trainer:
    """TensorFlow training engine for VisionMatch AI."""

    def __init__(
        self,
        model,
        loss_fn,
        config: TrainingConfig,
        optimizer=None,
        scheduler=None,
    ):
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance.")

        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = config.resolve_device()

        self.global_step = 0
        self.current_epoch = 0
        self.matching_batches_skipped = 0

        self.use_matching_loss = (
            bool(getattr(model, "use_cross_modal_attention", False))
            and config.matching_loss_weight > 0
        )

        self.matching_loss_fn = MatchingBinaryCrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "train_retrieval_loss": [],
            "train_matching_loss": [],
            "val_loss": [],
            "val_retrieval_loss": [],
            "val_matching_loss": [],
            "val_total_loss": [],
            "learning_rate": [],
        }

        self._ensure_model_built()

        self.optimizer = (
            optimizer
            if optimizer is not None
            else build_optimizer(self.model, self.config)
        )

        if hasattr(self.optimizer, "build"):
            self.optimizer.build(self.model.trainable_variables)
            self.model.optimizer = self.optimizer

        self.scheduler = (
            scheduler
            if scheduler is not None
            else build_scheduler(self.optimizer, self.config)
        )

        if self.scheduler is not None:
            self.scheduler.set_model(self.model)
            self.scheduler.set_params({
                "epochs": self.config.epochs,
                "steps": None,
                "verbose": 0,
                "metrics": ["val_loss"],
            })

        print("Trainer initialized successfully.")
        print(f"Device: {self.device}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"Cross-modal matching loss: {self.use_matching_loss}")

        if self.use_matching_loss:
            print(
                "Matching loss weight: "
                f"{self.config.matching_loss_weight}"
            )

        if self.scheduler is not None:
            print(f"Scheduler: {type(self.scheduler).__name__}")

    def _ensure_model_built(self):
        if self.model.built and self.model.trainable_variables:
            return

        dummy_images = tf.zeros(
            (1, 224, 224, 3),
            dtype=tf.float32,
        )
        dummy_captions = tf.fill(
            (1, 5, 34),
            tf.constant(4, dtype=tf.int32),
        )

        with tf.device(self.device):
            self.model(
                dummy_images,
                dummy_captions,
                training=False,
            )

        if not self.model.trainable_variables:
            raise RuntimeError("The model has no trainable variables.")

    @staticmethod
    def _prepare_batch(batch):
        if not isinstance(batch, dict):
            raise TypeError("Each batch must be a dictionary.")

        if "image" not in batch or "captions" not in batch:
            raise KeyError(
                "Batch must contain 'image' and 'captions'."
            )

        images = tf.convert_to_tensor(
            batch["image"],
            dtype=tf.float32,
        )
        captions = tf.convert_to_tensor(
            batch["captions"],
            dtype=tf.int32,
        )

        tf.debugging.assert_rank(images, 4)
        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must have shape [B,H,W,3].",
        )
        tf.debugging.assert_rank(captions, 3)
        tf.debugging.assert_equal(
            tf.shape(images)[0],
            tf.shape(captions)[0],
            message="Image and caption batch sizes must match.",
        )
        tf.debugging.assert_positive(tf.shape(images)[0])
        tf.debugging.assert_positive(tf.shape(captions)[1])

        return images, captions

    def _compute_loss_components(
        self,
        outputs,
        negative_matching_scores=None,
    ):
        if not isinstance(outputs, dict):
            raise TypeError("Model must return a dictionary.")

        required = {"image_embeddings", "text_embeddings"}
        missing = required - outputs.keys()
        if missing:
            raise KeyError(f"Missing model outputs: {missing}")

        retrieval_loss = tf.convert_to_tensor(
            self.loss_fn(
                outputs["image_embeddings"],
                outputs["text_embeddings"],
            )
        )
        if retrieval_loss.shape.rank != 0:
            raise ValueError("The retrieval loss function must return a scalar.")

        matching_loss = None
        total_loss = retrieval_loss

        if self.use_matching_loss and negative_matching_scores is not None:
            if "matching_scores" not in outputs:
                raise KeyError("Cross-modal model must return 'matching_scores'.")
            matching_loss = self.matching_loss_fn(
                outputs["matching_scores"],
                negative_matching_scores,
            )
            total_loss = (
                retrieval_loss
                + self.config.matching_loss_weight * matching_loss
            )
        elif self.use_matching_loss and negative_matching_scores is None:
            # Singleton batches have no valid in-batch negative.
            matching_loss = None

        tf.debugging.assert_all_finite(
            total_loss,
            "Non-finite total training loss detected.",
        )
        return {
            "total_loss": total_loss,
            "retrieval_loss": retrieval_loss,
            "matching_loss": matching_loss,
        }

    def _compute_loss(
        self,
        outputs,
        negative_matching_scores=None,
    ):
        """Backward-compatible scalar total objective."""
        return self._compute_loss_components(
            outputs,
            negative_matching_scores=negative_matching_scores,
        )["total_loss"]

    def _get_negative_matching_scores(
        self,
        images,
        captions,
        training,
    ):
        """
        Create in-batch negatives by cyclically shifting images.

        Each caption is paired with an image belonging to another
        sample in the same batch.
        """
        batch_size = int(tf.shape(images)[0].numpy())

        if batch_size < 2:
            self.matching_batches_skipped += 1
            return None

        negative_images = tf.roll(
            images,
            shift=1,
            axis=0,
        )

        negative_outputs = self.model(
            negative_images,
            captions,
            training=training,
        )

        if "matching_scores" not in negative_outputs:
            raise KeyError(
                "Model did not return negative matching scores."
            )

        return negative_outputs["matching_scores"]

    def train_epoch(self, train_loader):
        total_loss = 0.0
        retrieval_loss_sum = 0.0
        matching_loss_sum = 0.0
        matching_samples = 0
        total_samples = 0

        for batch_index, batch in enumerate(train_loader):
            images, captions = self._prepare_batch(batch)

            batch_size = int(tf.shape(images)[0].numpy())

            with tf.device(self.device):
                with tf.GradientTape() as tape:
                    outputs = self.model(
                        images,
                        captions,
                        training=True,
                    )

                    negative_scores = None

                    if self.use_matching_loss:
                        negative_scores = (
                            self._get_negative_matching_scores(
                                images,
                                captions,
                                training=True,
                            )
                        )

                    loss_components = self._compute_loss_components(
                        outputs,
                        negative_matching_scores=negative_scores,
                    )
                    loss = loss_components["total_loss"]
                    retrieval_loss = loss_components["retrieval_loss"]
                    matching_loss = loss_components["matching_loss"]

                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)

                gradient_pairs = [
                    (gradient, variable)
                    for gradient, variable in zip(gradients, variables)
                    if gradient is not None
                ]

                if not gradient_pairs:
                    raise RuntimeError(
                        "No gradients were produced. "
                        "Check the model and loss."
                    )

                grads, vars_with_gradients = zip(*gradient_pairs)

                if self.config.gradient_clip_norm is not None:
                    grads, _ = tf.clip_by_global_norm(
                        grads,
                        self.config.gradient_clip_norm,
                    )

                self.optimizer.apply_gradients(
                    zip(grads, vars_with_gradients)
                )

            total_loss += float(loss.numpy()) * batch_size
            retrieval_loss_sum += float(retrieval_loss.numpy()) * batch_size
            if matching_loss is not None:
                matching_loss_sum += float(matching_loss.numpy()) * batch_size
                matching_samples += batch_size
            total_samples += batch_size
            self.global_step += 1

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Batch [{batch_index + 1}] "
                    f"Loss: {float(loss.numpy()):.4f}"
                )

        if total_samples == 0:
            raise RuntimeError("The training dataset is empty.")

        average_loss = total_loss / total_samples
        average_retrieval_loss = retrieval_loss_sum / total_samples
        average_matching_loss = (
            matching_loss_sum / matching_samples
            if matching_samples > 0
            else None
        )
        self.history["train_loss"].append(average_loss)
        self.history["train_retrieval_loss"].append(
            average_retrieval_loss
        )
        self.history["train_matching_loss"].append(
            average_matching_loss
        )

        return average_loss

    def _compute_retrieval_loss(self, outputs):
        """Compute the original bi-encoder retrieval loss."""
        loss = self.loss_fn(
            outputs["image_embeddings"],
            outputs["text_embeddings"],
        )
        loss = tf.convert_to_tensor(loss)

        if loss.shape.rank != 0:
            raise ValueError(
                "The retrieval loss function must return a scalar."
            )

        tf.debugging.assert_all_finite(
            loss,
            "Non-finite retrieval loss detected.",
        )
        return loss

    def _get_learning_rate(self):
        learning_rate = self.optimizer.learning_rate

        if callable(learning_rate):
            learning_rate = learning_rate(
                self.optimizer.iterations
            )

        return float(
            tf.keras.backend.get_value(learning_rate)
        )

    @staticmethod
    def _get_early_stopping_state(early_stopping):
        if early_stopping is None:
            return None

        return {
            "best_loss": early_stopping.best_loss,
            "counter": early_stopping.counter,
            "early_stop": early_stopping.early_stop,
        }

    @staticmethod
    def _restore_early_stopping_state(early_stopping, state):
        if early_stopping is None or state is None:
            return

        early_stopping.best_loss = float(state["best_loss"])
        early_stopping.counter = int(state["counter"])
        early_stopping.early_stop = bool(state["early_stop"])

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
        total_epochs = (
            self.config.epochs if epochs is None else epochs
        )

        if (
            isinstance(total_epochs, bool)
            or not isinstance(total_epochs, int)
            or total_epochs <= 0
        ):
            raise ValueError("epochs must be a positive integer.")

        if self.config.scheduler == "plateau" and val_loader is None:
            raise ValueError(
                "Plateau scheduler requires validation data."
            )

        if checkpoint_manager is not None and val_loader is None:
            raise ValueError(
                "Checkpoint saving requires validation data."
            )

        if early_stopping is not None and val_loader is None:
            raise ValueError(
                "Early stopping requires validation data."
            )

        evaluator = (
            Evaluator(self.model, self.loss_fn, self.config)
            if val_loader is not None
            else None
        )

        if checkpoint_manager is None and resume_from is not None:
            checkpoint_manager = CheckpointManager(
                self.config.checkpoint_dir
            )

        start_epoch = 0

        if resume_from is not None:
            if checkpoint_manager is None:
                raise ValueError(
                    "A CheckpointManager is required when resuming."
                )

            loaded = checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )

            start_epoch = int(loaded["epoch"]) + 1
            self.current_epoch = start_epoch
            self.global_step = int(loaded["global_step"])

            self._restore_early_stopping_state(
                early_stopping,
                loaded.get("early_stopping_state"),
            )

            print(
                f"Checkpoint restored. "
                f"Resuming from epoch {start_epoch + 1}."
            )

        if start_epoch >= total_epochs:
            raise ValueError(
                "Checkpoint epoch has already reached "
                "the requested total epochs."
            )

        if self.scheduler is not None:
            self.scheduler.on_train_begin()

        print("\nStarting VisionMatch AI Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {total_epochs}")
        print(f"Starting Epoch: {start_epoch + 1}")
        print("-" * 55)

        try:
            for epoch in range(start_epoch, total_epochs):
                print(f"\nEpoch [{epoch + 1}/{total_epochs}]")

                if logger is not None:
                    logger.start_epoch()

                if self.scheduler is not None:
                    self.scheduler.on_epoch_begin(epoch)

                epoch_learning_rate = self._get_learning_rate()
                train_loss = self.train_epoch(train_loader)

                self.current_epoch = epoch + 1
                val_loss = None
                val_matching_loss = None
                val_total_loss = None

                if evaluator is not None:
                    validation_results = evaluator.evaluate(val_loader)
                    # val_loss is retrieval-only and is the shared
                    # model-selection metric across all architectures.
                    val_loss = float(validation_results["val_retrieval_loss"])
                    val_matching_loss = validation_results["val_matching_loss"]
                    val_total_loss = float(validation_results["val_total_loss"])

                    self.history["val_loss"].append(val_loss)
                    self.history["val_retrieval_loss"].append(val_loss)
                    self.history["val_matching_loss"].append(val_matching_loss)
                    self.history["val_total_loss"].append(val_total_loss)

                    print(f"Validation retrieval loss: {val_loss:.4f}")
                    if val_matching_loss is not None:
                        print(
                            "Validation matching loss (reported separately): "
                            f"{val_matching_loss:.4f}"
                        )

                self.history["learning_rate"].append(
                    epoch_learning_rate
                )

                logs = {
                    "loss": train_loss,
                    "val_loss": val_loss,
                }

                if self.scheduler is not None:
                    self.scheduler.on_epoch_end(epoch, logs=logs)

                should_stop = False
                if early_stopping is not None:
                    should_stop = early_stopping(val_loss)

                if checkpoint_manager is not None:
                    is_best = checkpoint_manager.update_best(val_loss)
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

                if logger is not None:
                    logger.log_epoch(
                        epoch=self.current_epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        learning_rate=epoch_learning_rate,
                        train_retrieval_loss=self.history["train_retrieval_loss"][-1],
                        train_matching_loss=self.history["train_matching_loss"][-1],
                        val_retrieval_loss=val_loss,
                        val_matching_loss=val_matching_loss,
                        val_total_loss=val_total_loss,
                    )

                print(f"Training objective: {train_loss:.4f}")
                print(
                    "Training retrieval loss: "
                    f"{self.history['train_retrieval_loss'][-1]:.4f}"
                )
                if self.history["train_matching_loss"][-1] is not None:
                    print(
                        "Training matching loss: "
                        f"{self.history['train_matching_loss'][-1]:.4f}"
                    )
                print(
                    f"Learning Rate Used: "
                    f"{epoch_learning_rate:.8f}"
                )
                print(
                    f"Next Learning Rate: "
                    f"{self._get_learning_rate():.8f}"
                )

                if should_stop:
                    print("\nEarly stopping: training terminated.")
                    break

        finally:
            if self.scheduler is not None:
                self.scheduler.on_train_end()

        print("\nTraining finished.")
        return self.history


if __name__ == "__main__":
    print("TensorFlow Trainer module imported successfully.")
