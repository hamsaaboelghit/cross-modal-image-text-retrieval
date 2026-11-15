
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.flickr30k_dataset import Flickr30kDataset
from src.models.full_model import CrossModalRetrievalModel
from src.training.contrastive_loss import MultiPositiveInfoNCELoss
from src.training.train_config import TrainingConfig
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def main():

    print("\n" + "=" * 60)
    print("VISIONMATCH AI - CONTINUE TRAINING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(
        "data/processed/flickr30k/vocabularies/fold_1_vocab.json",
        "r",
        encoding="utf-8",
    ) as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)

    train_dataset = Flickr30kDataset(1, "train")
    val_dataset = Flickr30kDataset(1, "validation")

    config = TrainingConfig(
        epochs=5,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        scheduler="none",
        loss_type="info_nce",
        temperature=0.07,
        use_amp=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
        seed=42,
        fold=1,
        device=device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=config.pin_memory,
    )

    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=False,
        unfreeze_last_block=False,
    )

    loss_fn = MultiPositiveInfoNCELoss(
        temperature=config.temperature,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    # Load Epoch 1 checkpoint
    checkpoint_path = Path(
        "checkpoints/real_fold1/epoch_1.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    trainer.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    trainer.current_epoch = 1
    trainer.global_step = 2543

    best_val_loss = checkpoint["val_loss"]

    print("\nResumed from Epoch 1")
    print(f"Previous validation loss: {best_val_loss:.6f}")

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    output_dir = Path("checkpoints/real_fold1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train Epochs 2 to 5
    for epoch in range(2, 6):

        print("\n" + "=" * 60)
        print(f"STARTING EPOCH {epoch}/5")
        print("=" * 60)

        start_time = time.time()

        train_loss = trainer.train_epoch(train_loader)

        validation = evaluator.evaluate(val_loader)
        val_loss = validation["val_loss"]

        epoch_time = (time.time() - start_time) / 60

        print(f"\nEpoch {epoch} Results")
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Time: {epoch_time:.2f} minutes")

        assert np.isfinite(train_loss)
        assert np.isfinite(val_loss)

        epoch_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "fold": 1,
            "model_type": "ResNet50-LSTM",
            "global_step": trainer.global_step,
        }

        torch.save(
            epoch_checkpoint,
            output_dir / f"epoch_{epoch}.pth",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                epoch_checkpoint,
                output_dir / "best_model.pth",
            )

            print("New best checkpoint saved!")

        print(f"Global steps: {trainer.global_step}")

    print("\n" + "=" * 60)
    print("TRAINING THROUGH EPOCH 5 COMPLETED")
    print("=" * 60)

    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints directory: {output_dir}")
    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
