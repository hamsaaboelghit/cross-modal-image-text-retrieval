
import csv
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.training.training_logger import TrainingLogger
from src.training.early_stopping import EarlyStopping


# =====================================================
# 1. Dummy Retrieval Model
# =====================================================

class DummyRetrievalModel(nn.Module):
    """
    Lightweight model for testing the complete fit pipeline.
    """

    def __init__(self, vocab_size=100, embedding_dim=32):
        super().__init__()

        self.image_projection = nn.Linear(
            3,
            embedding_dim,
        )

        self.text_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

    def forward(self, images, captions):

        # Image features: [B, 3, H, W] -> [B, 3]
        image_features = images.mean(dim=(2, 3))

        image_embeddings = self.image_projection(
            image_features
        )

        # Text features: [B, C, L] -> [B, C, L, D]
        text_features = self.text_embedding(captions)

        mask = (captions != 0).unsqueeze(-1)

        text_sum = (text_features * mask).sum(dim=2)

        token_count = mask.sum(dim=2).clamp(min=1)

        text_embeddings = text_sum / token_count

        image_embeddings = F.normalize(
            image_embeddings,
            dim=-1,
        )

        text_embeddings = F.normalize(
            text_embeddings,
            dim=-1,
        )

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }


# =====================================================
# 2. Dummy Retrieval Loss
# =====================================================

class DummyRetrievalLoss(nn.Module):
    """
    Symmetric image-text contrastive loss for testing.
    """

    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):

        # Average captions belonging to each image.
        text_embeddings = text_embeddings.mean(dim=1)

        logits = (
            image_embeddings @ text_embeddings.T
        ) / self.temperature

        labels = torch.arange(
            image_embeddings.size(0),
            device=image_embeddings.device,
        )

        image_loss = F.cross_entropy(
            logits,
            labels,
        )

        text_loss = F.cross_entropy(
            logits.T,
            labels,
        )

        return (image_loss + text_loss) / 2


# =====================================================
# 3. Synthetic Data Loader
# =====================================================

def create_dummy_loader(
    num_batches=3,
    batch_size=4,
    num_captions=5,
    sequence_length=12,
):
    """
    Create synthetic image-caption batches.
    """

    loader = []

    for _ in range(num_batches):

        images = torch.randn(
            batch_size,
            3,
            224,
            224,
        )

        captions = torch.randint(
            1,
            100,
            (
                batch_size,
                num_captions,
                sequence_length,
            ),
        )

        # Simulate padding.
        captions[:, :, -2:] = 0

        loader.append({
            "image": images,
            "captions": captions,
        })

    return loader


# =====================================================
# 4. Main Integration Test
# =====================================================

def run_fit_pipeline_tests():

    print("\n" + "=" * 60)
    print("VISIONMATCH AI - FULL FIT PIPELINE TEST")
    print("=" * 60)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------
    # Temporary directories
    # -------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        checkpoint_dir = temp_path / "checkpoints"
        log_dir = temp_path / "logs"

        # -------------------------------------------------
        # Configuration
        # -------------------------------------------------

        config = TrainingConfig(
            epochs=3,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            scheduler="cosine",
            loss_type="info_nce",
            use_amp=False,
            device=device,
            gradient_clip_norm=1.0,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
        )

        # -------------------------------------------------
        # Model and Loss
        # -------------------------------------------------

        model = DummyRetrievalModel()

        loss_fn = DummyRetrievalLoss()

        # -------------------------------------------------
        # Data
        # -------------------------------------------------

        train_loader = create_dummy_loader(
            num_batches=3,
            batch_size=4,
        )

        val_loader = create_dummy_loader(
            num_batches=2,
            batch_size=4,
        )

        # -------------------------------------------------
        # Training Components
        # -------------------------------------------------

        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
        )

        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name="fit_pipeline_test",
        )

        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
        )

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            config=config,
        )

        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        # =================================================
        # TEST 1: Complete fit()
        # =================================================

        print("\n[Test 1] Running Complete Training Pipeline")

        history = trainer.fit(
            train_loader=train_loader,
            epochs=3,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            early_stopping=early_stopping,
        )

        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["learning_rate"]) == 3

        print("PASS: Training completed for 3 epochs.")
        print("PASS: Training history contains all epochs.")

        # =================================================
        # TEST 2: Model Parameters Updated
        # =================================================

        print("\n[Test 2] Model Parameter Updates")

        parameters_updated = any(
            not torch.equal(
                initial_parameters[name],
                parameter.detach(),
            )
            for name, parameter in model.named_parameters()
        )

        assert parameters_updated

        print("PASS: Model parameters changed during training.")

        # =================================================
        # TEST 3: Checkpoint Files and Metadata
        # =================================================

        print("\n[Test 3] Checkpoint Files")

        latest_path = (
            checkpoint_dir / "latest_checkpoint.pth"
        )

        best_path = (
            checkpoint_dir / "best_model.pth"
        )

        assert latest_path.is_file()
        assert best_path.is_file()

        print("PASS: Latest checkpoint exists.")
        print("PASS: Best model checkpoint exists.")

        latest_checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=True,
        )

        assert latest_checkpoint["epoch"] == 2

        assert latest_checkpoint["global_step"] == 9

        assert torch.isfinite(
            torch.tensor(
                latest_checkpoint["best_val_loss"]
            )
        ).item()

        print("PASS: Checkpoint metadata is valid.")
        print(
            "PASS: Best validation loss = "
            f"{latest_checkpoint['best_val_loss']:.4f}"
        )

        # =================================================
        # TEST 4: Early Stopping Checkpoint State
        # =================================================

        print("\n[Test 4] Early Stopping Checkpoint State")

        saved_early_state = latest_checkpoint.get(
            "early_stopping_state"
        )

        assert saved_early_state is not None, (
            "Early stopping state was not saved."
        )

        assert saved_early_state["counter"] == (
            early_stopping.counter
        )

        assert saved_early_state["best_loss"] == (
            early_stopping.best_loss
        )

        assert saved_early_state["early_stop"] == (
            early_stopping.early_stop
        )

        assert saved_early_state["counter"] == early_stopping.counter

        assert saved_early_state["early_stop"] is False

        print("PASS: Early stopping state exists in checkpoint.")
        print("PASS: Saved counter matches current counter.")
        print("PASS: Saved best_loss matches current best_loss.")
        print("PASS: Saved early_stop flag is correct.")

        print(
            f"Saved Early Stopping Counter: "
            f"{saved_early_state['counter']}"
        )

        print(
            f"Saved Early Stopping Best Loss: "
            f"{saved_early_state['best_loss']:.6f}"
        )

        # =================================================
        # TEST 5: Checkpoint Restore
        # =================================================

        print("\n[Test 5] Checkpoint Restore")

        restored_model = DummyRetrievalModel()

        restored_trainer = Trainer(
            model=restored_model,
            loss_fn=DummyRetrievalLoss(),
            config=config,
        )

        restore_info = checkpoint_manager.load_checkpoint(
            checkpoint_path=latest_path,
            model=restored_trainer.model,
            optimizer=restored_trainer.optimizer,
            scheduler=restored_trainer.scheduler,
            map_location=device,
        )

        assert restore_info["epoch"] == 2
        assert restore_info["global_step"] == 9

        for name, parameter in model.named_parameters():

            restored_parameter = dict(
                restored_model.named_parameters()
            )[name]

            assert torch.equal(
                parameter.detach(),
                restored_parameter.detach(),
            )

        print("PASS: Checkpoint metadata restored.")
        print("PASS: Model parameters restored.")

        # =================================================
        # TEST 6: Early Stopping Resume
        # =================================================

        print("\n[Test 6] Early Stopping Resume")

        restored_early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
        )

        restored_trainer._restore_early_stopping_state(
            restored_early_stopping,
            restore_info.get("early_stopping_state"),
        )

        assert (
            restored_early_stopping.counter
            == early_stopping.counter
        )

        assert (
            restored_early_stopping.best_loss
            == early_stopping.best_loss
        )

        assert (
            restored_early_stopping.early_stop
            == early_stopping.early_stop
        )

        print("PASS: Early stopping counter restored.")
        print("PASS: Early stopping best loss restored.")
        print("PASS: Early stopping flag restored.")

        # =================================================
        # TEST 7: Training CSV Log
        # =================================================

        print("\n[Test 7] Training Logger")

        csv_files = list(log_dir.glob("*.csv"))

        assert len(csv_files) == 1

        with open(
            csv_files[0],
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            rows = list(csv.DictReader(file))

        assert len(rows) == 3

        required_columns = {
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
            "epoch_duration",
            "timestamp",
        }

        assert required_columns.issubset(
            rows[0].keys()
        )

        print("PASS: CSV log file exists.")
        print("PASS: All 3 epochs were logged.")

        # =================================================
        # TEST 8: History Validation
        # =================================================

        print("\n[Test 8] Training History")

        for value in history["train_loss"]:
            assert torch.isfinite(
                torch.tensor(value)
            ).item()

        for value in history["val_loss"]:
            assert torch.isfinite(
                torch.tensor(value)
            ).item()

        for value in history["learning_rate"]:
            assert value > 0

        print("PASS: All training losses are finite.")
        print("PASS: All validation losses are finite.")
        print("PASS: Learning rates are valid.")

        # =================================================
        # TEST 9: Global Step and Epoch Count
        # =================================================

        print("\n[Test 9] Training Progress")

        expected_global_step = (
            len(train_loader) * 3
        )

        assert trainer.global_step == expected_global_step
        assert trainer.current_epoch == 3

        print(
            f"PASS: Global step = {trainer.global_step}"
        )

        print(
            f"PASS: Current epoch = {trainer.current_epoch}"
        )

        # =================================================
        # Final Results
        # =================================================

        print("\n" + "=" * 60)
        print("ALL 9 FULL PIPELINE TESTS PASSED!")
        print("=" * 60)

        print(f"Device: {device}")

        print(
            f"Final Training Loss: "
            f"{history['train_loss'][-1]:.4f}"
        )

        print(
            f"Final Validation Loss: "
            f"{history['val_loss'][-1]:.4f}"
        )

        print(
            f"Total Training Steps: "
            f"{trainer.global_step}"
        )

        print("\nNote: Synthetic data was used.")
        print("These are NOT real Flickr30k results.")


if __name__ == "__main__":

    run_fit_pipeline_tests()
