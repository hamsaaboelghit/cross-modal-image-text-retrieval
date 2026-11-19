import tensorflow as tf

from src.training.train_config import TrainingConfig
from src.training.matching_loss import MatchingBinaryCrossEntropyLoss


class Evaluator:
    """Validation engine with retrieval-only model-selection loss."""

    def __init__(self, model, loss_fn, config: TrainingConfig):
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance.")

        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = config.resolve_device()

        self.use_matching_loss = (
            bool(getattr(model, "use_cross_modal_attention", False))
            and config.matching_loss_weight > 0
        )
        self.matching_loss_fn = MatchingBinaryCrossEntropyLoss()
        self.matching_batches_skipped = 0

        print("Evaluator initialized successfully.")
        print(f"Device: {self.device}")
        print("Model-selection metric: retrieval loss only")
        print(f"Cross-modal matching loss: {self.use_matching_loss}")

    @staticmethod
    def _prepare_batch(batch):
        if not isinstance(batch, dict):
            raise TypeError("Each batch must be a dictionary.")

        if "image" not in batch or "captions" not in batch:
            raise KeyError("Batch must contain 'image' and 'captions'.")

        images = tf.convert_to_tensor(batch["image"], dtype=tf.float32)
        captions = tf.convert_to_tensor(batch["captions"], dtype=tf.int32)

        tf.debugging.assert_rank(images, 4)
        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must use channels-last format [B,H,W,3].",
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

    def _compute_retrieval_loss(self, outputs):
        if not isinstance(outputs, dict):
            raise TypeError("Model must return a dictionary.")

        required = {"image_embeddings", "text_embeddings"}
        missing = required - outputs.keys()
        if missing:
            raise KeyError(f"Missing model outputs: {missing}")

        loss = tf.convert_to_tensor(
            self.loss_fn(
                outputs["image_embeddings"],
                outputs["text_embeddings"],
                training=False,
            )
        )

        if loss.shape.rank != 0:
            raise ValueError("The retrieval loss function must return a scalar.")

        tf.debugging.assert_all_finite(
            loss,
            "Non-finite retrieval validation loss detected.",
        )
        return loss

    def evaluate(self, val_loader, max_batches=None):
        if max_batches is not None and (
            isinstance(max_batches, bool)
            or not isinstance(max_batches, int)
            or max_batches <= 0
        ):
            raise ValueError("max_batches must be a positive integer or None.")

        retrieval_loss_sum = 0.0
        matching_loss_sum = 0.0
        total_samples = 0
        matching_samples = 0
        total_batches = 0
        matching_batches_skipped = 0

        for batch_index, batch in enumerate(val_loader):
            images, captions = self._prepare_batch(batch)
            batch_size = int(tf.shape(images)[0].numpy())

            with tf.device(self.device):
                outputs = self.model(images, captions, training=False)
                retrieval_loss = self._compute_retrieval_loss(outputs)

                matching_loss = None
                if self.use_matching_loss:
                    if batch_size < 2:
                        matching_batches_skipped += 1
                    else:
                        negative_images = tf.roll(images, shift=1, axis=0)
                        negative_outputs = self.model(
                            negative_images,
                            captions,
                            training=False,
                        )
                        if "matching_scores" not in outputs or "matching_scores" not in negative_outputs:
                            raise KeyError(
                                "Cross-modal model must return 'matching_scores'."
                            )

                        matching_loss = self.matching_loss_fn(
                            outputs["matching_scores"],
                            negative_outputs["matching_scores"],
                        )
                        tf.debugging.assert_all_finite(
                            matching_loss,
                            "Non-finite matching validation loss detected.",
                        )

            retrieval_loss_sum += float(retrieval_loss.numpy()) * batch_size
            total_samples += batch_size
            total_batches += 1

            if matching_loss is not None:
                matching_loss_sum += float(matching_loss.numpy()) * batch_size
                matching_samples += batch_size

            if max_batches is not None and batch_index + 1 >= max_batches:
                break

        if total_samples == 0:
            raise RuntimeError("The validation dataset is empty.")

        val_retrieval_loss = retrieval_loss_sum / total_samples
        val_matching_loss = (
            matching_loss_sum / matching_samples
            if matching_samples > 0
            else None
        )

        val_total_loss = val_retrieval_loss
        if val_matching_loss is not None:
            val_total_loss += (
                self.config.matching_loss_weight * val_matching_loss
            )

        self.matching_batches_skipped += matching_batches_skipped

        return {
            # Keep val_loss as a backwards-compatible alias. It is now
            # deliberately retrieval-only for every architecture.
            "val_loss": val_retrieval_loss,
            "val_retrieval_loss": val_retrieval_loss,
            "val_matching_loss": val_matching_loss,
            "val_total_loss": val_total_loss,
            "num_samples": total_samples,
            "num_batches": total_batches,
            "matching_batches_skipped": matching_batches_skipped,
        }


if __name__ == "__main__":
    print("TensorFlow Evaluator module imported successfully.")
