import tensorflow as tf

from src.training.train_config import TrainingConfig


class Evaluator:
    """TensorFlow validation engine for VisionMatch AI."""

    def __init__(self, model, loss_fn, config: TrainingConfig):
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance.")

        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.device = config.resolve_device()

        print("Evaluator initialized successfully.")
        print(f"Device: {self.device}")

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
            tf.shape(images)[-1], 3,
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

    def evaluate(self, val_loader):
        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        for batch in val_loader:
            images, captions = self._prepare_batch(batch)

            with tf.device(self.device):
                outputs = self.model(
                    images,
                    captions,
                    training=False,
                )

                if not isinstance(outputs, dict):
                    raise TypeError("Model must return a dictionary.")

                required = {"image_embeddings", "text_embeddings"}
                missing = required - outputs.keys()

                if missing:
                    raise KeyError(f"Missing model outputs: {missing}")

                loss = self.loss_fn(
                    outputs["image_embeddings"],
                    outputs["text_embeddings"],
                )

                loss = tf.convert_to_tensor(loss)

                if loss.shape.rank != 0:
                    raise ValueError("The loss function must return a scalar.")

                tf.debugging.assert_all_finite(
                    loss,
                    "Non-finite validation loss detected.",
                )

            batch_size = int(tf.shape(images)[0].numpy())

            total_loss += float(loss.numpy()) * batch_size
            total_samples += batch_size
            total_batches += 1

        if total_samples == 0:
            raise RuntimeError("The validation dataset is empty.")

        return {
            "val_loss": total_loss / total_samples,
            "num_samples": total_samples,
            "num_batches": total_batches,
        }


if __name__ == "__main__":
    print("TensorFlow Evaluator module imported successfully.")