
import json
import random
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
    print("VISIONMATCH AI - FULL FLICKR30K TRAINING")
    print("=" * 60)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vocab_path = Path(
        "data/processed/flickr30k/"
        "vocabularies/fold_1_vocab.json"
    )

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    vocab_size = len(vocabulary)
    print(f"Vocabulary size: {vocab_size}")

    train_dataset = Flickr30kDataset(
        fold_number=1,
        split="train",
    )

    val_dataset = Flickr30kDataset(
        fold_number=1,
        split="validation",
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    config = TrainingConfig(
        epochs=1,
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
        seed=seed,
        fold=1,
        device=device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=config.pin_memory,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Mixed precision requested: {config.use_amp}")

    print("\nInitializing ResNet50 + LSTM...")

    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=True,
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

    print("\nStarting full training epoch...")
    start_time = time.time()

    train_loss = trainer.train_epoch(train_loader)

    training_time = time.time() - start_time

    print(f"\nTraining loss: {train_loss:.6f}")
    print(f"Training time: {training_time / 60:.2f} minutes")

    print("\nStarting full validation...")

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    validation_results = evaluator.evaluate(val_loader)
    val_loss = validation_results["val_loss"]

    print(f"Validation loss: {val_loss:.6f}")

    checkpoint_dir = Path("checkpoints/real_fold1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "epoch_1.pth"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "fold": 1,
            "model_type": "ResNet50-LSTM",
        },
        checkpoint_path,
    )

    total_time = time.time() - start_time

    assert np.isfinite(train_loss)
    assert np.isfinite(val_loss)

    print("\n" + "=" * 60)
    print("FULL FLICKR30K TRAINING COMPLETED")
    print("=" * 60)
    print(f"Training loss: {train_loss:.6f}")
    print(f"Validation loss: {val_loss:.6f}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Training batches: {trainer.global_step}")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Checkpoint: {checkpoint_path}")

    if device == "cuda":
        print(
            "Peak GPU memory: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )

    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
