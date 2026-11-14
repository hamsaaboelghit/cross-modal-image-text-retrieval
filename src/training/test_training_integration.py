
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.train_config import TrainingConfig
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.training.checkpoint_manager import CheckpointManager
from src.training.training_logger import TrainingLogger
from src.training.early_stopping import EarlyStopping


# =====================================================
# 1. Dummy Model
# =====================================================

class DummyRetrievalModel(nn.Module):
    """
    Small model for integration testing.

    It simulates the expected VisionMatch AI outputs:
        image_embeddings: [B, D]
        text_embeddings:  [B, C, D]
    """

    def __init__(self, vocab_size=100, embedding_dim=32):
        super().__init__()

        self.image_projection = nn.Linear(3, embedding_dim)

        self.text_embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

    def forward(self, images, captions):

        # [B, 3, H, W] -> [B, 3]
        image_features = images.mean(dim=(2, 3))

        image_embeddings = self.image_projection(
            image_features
        )

        # [B, C, L] -> [B, C, L, D]
        text_features = self.text_embedding(captions)

        # Ignore padding tokens.
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
    Symmetric image-text contrastive loss.

    Used only for testing the training pipeline.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):

        # Average the five captions for each image.
        text_embeddings = text_embeddings.mean(dim=1)

        logits = (
            image_embeddings @ text_embeddings.T
        ) / self.temperature

        labels = torch.arange(
            image_embeddings.size(0),
            device=image_embeddings.device,
        )

        image_loss = F.cross_entropy(logits, labels)

        text_loss = F.cross_entropy(logits.T, labels)

        return (image_loss + text_loss) / 2


# =====================================================
# 3. Test Data
# =====================================================

def create_dummy_loader(
    num_batches=3,
    batch_size=4,
    num_captions=5,
    sequence_length=12,
):
    """
    Generate synthetic image-caption batches.
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

        # Add padding to some caption positions.
        captions[:, :, -2:] = 0

        loader.append({
            "image": images,
            "captions": captions,
        })

    return loader


# =====================================================
# 4. Integration Tests
# =====================================================

def run_integration_tests():

    print("\n" + "=" * 55)
    print("VISIONMATCH AI - TRAINING INTEGRATION TESTS")
    print("=" * 55)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = TrainingConfig(
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        loss_type="info_nce",
        use_amp=False,
        device=device,
        gradient_clip_norm=1.0,
    )

    model = DummyRetrievalModel()
    loss_fn = DummyRetrievalLoss()

    train_loader = create_dummy_loader()
    val_loader = create_dummy_loader(
        num_batches=2
    )

    # -------------------------------------------------
    # Test 1: Optimizer
    # -------------------------------------------------

    print("\n[Test 1] Optimizer Creation")

    optimizer = build_optimizer(model, config)

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )

    print("PASS: AdamW optimizer created.")

    # -------------------------------------------------
    # Test 2: Scheduler
    # -------------------------------------------------

    print("\n[Test 2] Scheduler Creation")

    scheduler = build_scheduler(
        optimizer,
        config,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.CosineAnnealingLR,
    )

    print("PASS: Cosine scheduler created.")

    # -------------------------------------------------
    # Test 3: Trainer Initialization
    # -------------------------------------------------

    print("\n[Test 3] Trainer Initialization")

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert trainer.device.type == device

    print("PASS: Trainer initialized.")

    # -------------------------------------------------
    # Test 4: Training Epoch
    # -------------------------------------------------

    print("\n[Test 4] Training Epoch")

    params_before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    train_loss = trainer.train_epoch(train_loader)

    assert isinstance(train_loss, float)
    assert torch.isfinite(torch.tensor(train_loss))

    params_after = list(model.parameters())

    weights_updated = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            params_before,
            params_after,
        )
    )

    assert weights_updated

    assert trainer.global_step == len(train_loader)

    print(f"PASS: Training loss = {train_loss:.4f}")
    print("PASS: Model parameters were updated.")

    # -------------------------------------------------
    # Test 5: Validation
    # -------------------------------------------------

    print("\n[Test 5] Validation")

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    validation_params_before = [
        p.detach().clone()
        for p in model.parameters()
    ]

    validation_results = evaluator.evaluate(
        val_loader
    )

    validation_params_after = list(model.parameters())

    assert torch.isfinite(
        torch.tensor(validation_results["val_loss"])
    )

    assert validation_results["num_samples"] == 8
    assert validation_results["num_batches"] == 2

    weights_unchanged = all(
        torch.equal(before, after.detach())
        for before, after in zip(
            validation_params_before,
            validation_params_after,
        )
    )

    assert weights_unchanged

    print(
        f"PASS: Validation loss = "
        f"{validation_results['val_loss']:.4f}"
    )

    print("PASS: Validation did not update model weights.")

    # -------------------------------------------------
    # Test 6: Scheduler Step
    # -------------------------------------------------

    print("\n[Test 6] Scheduler Update")

    trainer.step_scheduler()

    print("PASS: Scheduler updated.")

    # -------------------------------------------------
    # Test 7: Checkpoint Save and Load
    # -------------------------------------------------

    print("\n[Test 7] Checkpoint Save and Load")

    with tempfile.TemporaryDirectory() as temp_dir:

        manager = CheckpointManager(
            checkpoint_dir=temp_dir
        )

        saved_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            val_loss=validation_results["val_loss"],
            global_step=trainer.global_step,
            config=config,
            is_best=True,
        )

        assert Path(saved_path).is_file()

        saved_model_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

        # Modify model parameters before loading.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)

        loaded_info = manager.load_checkpoint(
            checkpoint_path=saved_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )

        assert loaded_info["epoch"] == 1

        for key, value in model.state_dict().items():
            assert torch.equal(
                value,
                saved_model_state[key],
            )

        assert Path(temp_dir, "best_model.pth").is_file()

        print("PASS: Checkpoint saved.")
        print("PASS: Model weights restored.")
        print("PASS: Best model checkpoint saved.")

    # -------------------------------------------------
    # Test 8: Training Logger
    # -------------------------------------------------

    print("\n[Test 8] Training Logger")

    with tempfile.TemporaryDirectory() as temp_dir:

        logger = TrainingLogger(
            log_dir=temp_dir,
            experiment_name="integration_test",
        )

        logger.start_epoch()

        logger.log_epoch(
            epoch=1,
            train_loss=train_loss,
            val_loss=validation_results["val_loss"],
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        assert logger.get_log_path().is_file()

        with open(
            logger.get_log_path(),
            encoding="utf-8",
        ) as file:
            lines = file.readlines()

        assert len(lines) == 2

        print("PASS: Training metrics saved to CSV.")

    # -------------------------------------------------
    # Test 9: Early Stopping
    # -------------------------------------------------

    print("\n[Test 9] Early Stopping")

    early_stopping = EarlyStopping(
        patience=2,
        min_delta=0.001,
    )

    assert early_stopping(0.90) is False
    assert early_stopping(0.85) is False
    assert early_stopping(0.86) is False

    should_stop = early_stopping(0.87)

    assert should_stop is True

    print("PASS: Early stopping triggered correctly.")

    # -------------------------------------------------
    # Final Results
    # -------------------------------------------------

    print("\n" + "=" * 55)
    print("ALL 9 INTEGRATION TESTS PASSED!")
    print("=" * 55)

    print(f"Device: {device}")
    print(f"Training Loss: {train_loss:.4f}")
    print(
        f"Validation Loss: "
        f"{validation_results['val_loss']:.4f}"
    )

    print("\nNote: All values above are from synthetic test data.")
    print("They are NOT real VisionMatch AI performance results.")


if __name__ == "__main__":
    run_integration_tests()
