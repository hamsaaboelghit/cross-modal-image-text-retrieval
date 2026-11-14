
import torch
from torch import nn

from src.training.train_config import TrainingConfig


class Evaluator:
    """
    Validation engine for VisionMatch AI.

    Responsibilities:
        - Evaluate the model without updating weights.
        - Calculate average validation loss.
        - Track validation samples and batches.
        - Detect invalid loss values.

    Expected batch:
        image:    [B, 3, H, W]
        captions: [B, C, L]
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
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

        self.use_amp = (
            config.use_amp and self.device.type == "cuda"
        )

        print("Evaluator initialized successfully.")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")

    def _prepare_batch(self, batch):
        """Move validation data to the selected device."""

        if not isinstance(batch, dict):
            raise TypeError(
                "Each batch must be a dictionary."
            )

        if "image" not in batch or "captions" not in batch:
            raise KeyError(
                "Batch must contain 'image' and 'captions'."
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

        if captions.ndim != 3:
            raise ValueError(
                f"Expected captions [B, C, L], got {captions.shape}."
            )

        if images.size(0) != captions.size(0):
            raise ValueError(
                "Image and caption batch sizes must match."
            )

        return images, captions

    def evaluate(self, val_loader):
        """
        Evaluate the model on the complete validation dataset.

        Returns:
            Average validation loss.
        """

        self.model.eval()
        self.loss_fn.eval()

        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        with torch.no_grad():

            for batch in val_loader:

                images, captions = self._prepare_batch(batch)

                batch_size = images.size(0)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(images, captions)

                    if not isinstance(outputs, dict):
                        raise TypeError(
                            "Model must return a dictionary."
                        )

                    if (
                        "image_embeddings" not in outputs
                        or "text_embeddings" not in outputs
                    ):
                        raise KeyError(
                            "Model outputs must contain "
                            "'image_embeddings' and 'text_embeddings'."
                        )

                    loss = self.loss_fn(
                        outputs["image_embeddings"],
                        outputs["text_embeddings"],
                    )

                if loss.ndim != 0:
                    raise ValueError(
                        "The loss function must return a scalar."
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss: {loss.item()}"
                    )

                total_loss += loss.item() * batch_size
                total_samples += batch_size
                total_batches += 1

        if total_samples == 0:
            raise RuntimeError(
                "The validation DataLoader is empty."
            )

        average_loss = total_loss / total_samples

        return {
            "val_loss": average_loss,
            "num_samples": total_samples,
            "num_batches": total_batches,
        }


if __name__ == "__main__":
    print("Evaluator module imported successfully.")
    print("Use Evaluator(model, loss_fn, config) for validation.")
