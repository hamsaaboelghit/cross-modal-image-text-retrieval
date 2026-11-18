
import tensorflow as tf

from src.training.train_config import TrainingConfig
from src.training.matching_loss import MatchingBinaryCrossEntropyLoss


class Evaluator:
    """TensorFlow validation engine for VisionMatch AI."""

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
        print(f"Cross-modal matching loss: {self.use_matching_loss}")

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

    def _compute_loss(
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

        retrieval_loss = self.loss_fn(
            outputs["image_embeddings"],
            outputs["text_embeddings"],
        )
        retrieval_loss = tf.convert_to_tensor(retrieval_loss)

        if retrieval_loss.shape.rank != 0:
            raise ValueError(
                "The retrieval loss function must return a scalar."
            )

        total_loss = retrieval_loss

        if self.use_matching_loss:
            if "matching_scores" not in outputs:
                raise KeyError(
                    "Cross-modal model must return 'matching_scores'."
                )

            if negative_matching_scores is None:
                raise ValueError(
                    "Negative matching scores are required "
                    "when matching loss is enabled."
                )

            matching_loss = self.matching_loss_fn(
                outputs["matching_scores"],
                negative_matching_scores,
            )

            total_loss = (
                retrieval_loss
                + self.config.matching_loss_weight * matching_loss
            )

        tf.debugging.assert_all_finite(
            total_loss,
            "Non-finite validation loss detected.",
        )

        return total_loss

    def evaluate(self, val_loader):
        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        for batch in val_loader:
            images, captions = self._prepare_batch(batch)
            batch_size = int(tf.shape(images)[0].numpy())

            with tf.device(self.device):
                outputs = self.model(
                    images,
                    captions,
                    training=False,
                )

                negative_scores = None

                if self.use_matching_loss:
                    if batch_size < 2:
                        self.matching_batches_skipped += 1
                    else:
                        negative_images = tf.roll(
                            images,
                            shift=1,
                            axis=0,
                        )
                        negative_outputs = self.model(
                            negative_images,
                            captions,
                            training=False,
                        )
                        negative_scores = (
                            negative_outputs["matching_scores"]
                        )

                if self.use_matching_loss and negative_scores is None:
                    loss = self.loss_fn(
                        outputs["image_embeddings"],
                        outputs["text_embeddings"],
                    )
                else:
                    loss = self._compute_loss(
                        outputs,
                        negative_matching_scores=negative_scores,
                    )

                loss = tf.convert_to_tensor(loss)

                if loss.shape.rank != 0:
                    raise ValueError(
                        "The validation loss must be a scalar."
                    )

                tf.debugging.assert_all_finite(
                    loss,
                    "Non-finite validation loss detected.",
                )

            total_loss += float(loss.numpy()) * batch_size
            total_samples += batch_size
            total_batches += 1

        if total_samples == 0:
            raise RuntimeError("The validation dataset is empty.")

        return {
            "val_loss": total_loss / total_samples,
            "num_samples": total_samples,
            "num_batches": total_batches,
            "matching_batches_skipped": self.matching_batches_skipped,
        }


if __name__ == "__main__":
    print("TensorFlow Evaluator module imported successfully.")
