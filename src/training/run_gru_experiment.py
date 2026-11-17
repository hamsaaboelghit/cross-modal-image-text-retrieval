
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed = 42
    fold = 1

    # Shared visual encoder configuration
    pretrained = True
    unfreeze_last_block = False

    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("VISIONMATCH AI - GRU EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    vocab_path = Path(
        f"data/processed/flickr30k/vocabularies/fold_{fold}_vocab.json"
    )

    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}"
        )

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    vocab_size = len(vocabulary)

    train_dataset = Flickr30kDataset(fold, "train")
    val_dataset = Flickr30kDataset(fold, "validation")

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
        seed=seed,
        fold=fold,
        device=device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    model_config = {
        "encoder_type": "gru",
        "embedding_dim": 300,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "pretrained": pretrained,
        "unfreeze_last_block": unfreeze_last_block,
        "use_cross_modal_attention": False,
    }

    print("\nModel Configuration:")
    for key, value in model_config.items():
        print(f"{key}: {value}")

    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        **model_config,
    )

    loss_fn = MultiPositiveInfoNCELoss(
        temperature=config.temperature,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    output_dir = Path(
        f"checkpoints/experiments/gru_fold{fold}_seed{seed}_pretrained"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        print("\n" + "=" * 60)
        print(f"GRU - EPOCH {epoch}/{config.epochs}")
        print("=" * 60)

        start_time = time.time()

        train_loss = trainer.train_epoch(train_loader)

        validation = evaluator.evaluate(val_loader)
        val_loss = validation["val_loss"]

        elapsed = (time.time() - start_time) / 60

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Time: {elapsed:.2f} minutes")

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(
                "Non-finite loss detected. Experiment stopped."
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "fold": fold,
            "model_type": "ResNet50-GRU",
            "model_config": model_config,
            "training_config": config.to_dict(),
            "encoder_type": "gru",
            "seed": seed,
            "global_step": trainer.global_step,
        }

        torch.save(
            checkpoint,
            output_dir / f"epoch_{epoch}.pth",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                checkpoint,
                output_dir / "best_model.pth",
            )

            print("New best checkpoint saved!")

    print("\n" + "=" * 60)
    print("GRU EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved to: {output_dir}")
    print("Test split was NOT used.")


if __name__ == "__main__":
    main()
