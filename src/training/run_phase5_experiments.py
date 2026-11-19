"""Run the controlled Phase 5 data and initialization experiment matrix."""
import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import tensorflow as tf

from src.data.flickr30k_dataset import Flickr30kDataset
from src.data.vocabulary import load_vocabulary
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.models.full_model import CrossModalRetrievalModel
from src.training.experiment_runner import run_experiment


ROOT = Path(__file__).resolve().parents[2]
FOLD, SEED = 1, 42
LR, OUTPUT_DIM, DROPOUT = 1e-4, 512, 0.1
OUT = ROOT / "results" / "phase5_data_strategy"
VOCAB = ROOT / "data" / "processed" / "flickr30k" / "vocabularies" / "fold_1_vocab.json"


def build_experiments():
    base = {
        "model_type": "full", "encoder_type": "lstm",
        "use_cross_modal_attention": False, "unfreeze_last_block": False,
        "caption_count": 5, "image_augmentation": "none",
        "negative_sampling_strategy": "random", "pretrained": True,
        "train_all_backbone": False, "loss_type": "info_nce", "margin": None,
    }
    specs = [
        ("data_reference", {}),
        ("one_caption", {"caption_count": 1}),
        ("light_augmentation", {"image_augmentation": "light"}),
        ("triplet_random", {"loss_type": "triplet"}),
        ("triplet_hardest", {"loss_type": "triplet", "negative_sampling_strategy": "hardest"}),
        ("resnet50_pretrained_finetune", {"train_all_backbone": True}),
        ("resnet50_scratch_finetune", {"pretrained": False, "train_all_backbone": True}),
    ]
    return [{"name": name, **base, **changes} for name, changes in specs]


def metadata(all_experiments, epochs, batch_size, train_steps_per_epoch, validation_steps):
    return {
        "schema_version": 3, "fold": FOLD, "seed": SEED,
        "epochs": epochs, "batch_size": batch_size,
        "train_steps_per_epoch": train_steps_per_epoch,
        "validation_steps": validation_steps, "learning_rate": LR,
        "output_dim": OUTPUT_DIM, "dropout": DROPOUT,
        "selection_split": "validation", "test_set_used": False,
        "experiments": all_experiments,
    }


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.flush()
    temp.replace(path)


def evaluate_checkpoint(config, batch_size):
    vocab = load_vocabulary(VOCAB)
    model = CrossModalRetrievalModel(
        vocab_size=len(vocab), encoder_type="lstm", embedding_dim=300,
        hidden_dim=512, output_dim=OUTPUT_DIM, num_layers=1,
        dropout=DROPOUT, bidirectional=False, use_attention=True,
        pretrained=config["pretrained"], unfreeze_last_block=False,
        train_all_backbone=config["train_all_backbone"],
        use_cross_modal_attention=False,
    )
    model(
        tf.zeros((1, 224, 224, 3), tf.float32),
        tf.fill((1, 5, 34), tf.constant(4, tf.int32)),
        training=False,
    )
    checkpoint_dir = ROOT / "checkpoints" / "experiments" / config["name"] / "best"
    checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if checkpoint is None:
        raise FileNotFoundError(f"No best checkpoint in {checkpoint_dir}")
    status = tf.train.Checkpoint(model=model).restore(checkpoint)
    status.assert_existing_objects_matched()
    status.expect_partial()

    val = Flickr30kDataset(FOLD, "validation")
    val_ds = val.as_dataset(
        batch_size=batch_size, shuffle=False, drop_remainder=False, seed=SEED
    )
    metrics = RetrievalEvaluator(
        model, query_batch_size=32, candidate_chunk_size=8192, ks=(1, 5, 10)
    ).evaluate(val_ds)
    metrics["checkpoint_path"] = checkpoint
    return metrics


def save_results(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "phase5_results.json", rows)
    fields = [
        "experiment", "status", "batch_size", "epochs_requested",
        "train_steps_per_epoch", "validation_steps",
        "validation_loss_is_partial", "loss_type", "caption_count",
        "image_augmentation", "negative_sampling_strategy", "pretrained",
        "train_all_backbone", "epochs_completed", "best_val_retrieval_loss",
        "final_val_retrieval_loss", "training_time_minutes",
        "overall_validation_mean_recall", "text_to_image_r1",
        "text_to_image_r5", "text_to_image_r10", "image_to_text_r1",
        "image_to_text_r5", "image_to_text_r10", "text_to_image_mrr",
        "image_to_text_mrr", "checkpoint_path", "error",
    ]
    path, temp = OUT / "phase5_comparison.csv", OUT / "phase5_comparison.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
    temp.replace(path)


def run_all(
    experiments, all_experiments, epochs, batch_size,
    train_steps_per_epoch, validation_steps, resume, overwrite
):
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path, results_path = OUT / "phase5_run_metadata.json", OUT / "phase5_results.json"
    expected = metadata(
        all_experiments, epochs, batch_size,
        train_steps_per_epoch, validation_steps,
    )
    if resume:
        if not meta_path.is_file() or not results_path.is_file():
            raise FileNotFoundError("Resume needs existing Phase 5 metadata and results.")
        if json.loads(meta_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Saved Phase 5 settings differ; use matching settings or --overwrite.")
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    else:
        if (meta_path.exists() or results_path.exists()) and not overwrite:
            raise FileExistsError("Phase 5 results exist; use --resume or --overwrite.")
        rows = []
        atomic_json(meta_path, expected)

    completed = {
        r["experiment"] for r in rows
        if r.get("status") == "completed" and "overall_validation_mean_recall" in r
    }
    for i, config in enumerate(experiments, 1):
        name = config["name"]
        if name in completed:
            print(f"[SKIP] {name}")
            continue
        rows = [r for r in rows if r.get("experiment") != name]
        started = time.time()
        print(f"\nPHASE 5 {i}/{len(experiments)}: {name}")
        try:
            history = run_experiment(
                experiment_name=name, model_type=config["model_type"],
                encoder_type=config["encoder_type"],
                use_cross_modal_attention=config["use_cross_modal_attention"],
                pretrained=config["pretrained"],
                unfreeze_last_block=config["unfreeze_last_block"],
                train_all_backbone=config["train_all_backbone"],
                caption_count=config["caption_count"],
                image_augmentation=config["image_augmentation"],
                negative_sampling_strategy=config["negative_sampling_strategy"],
                epochs=epochs, batch_size=batch_size, seed=SEED, fold=FOLD,
                learning_rate=LR, loss_type=config["loss_type"],
                margin=config["margin"], output_dim=OUTPUT_DIM,
                dropout=DROPOUT, log_dir="runs/phase5_data_strategy",
                max_train_steps_per_epoch=train_steps_per_epoch,
                max_validation_steps=validation_steps,
            )
            metrics = evaluate_checkpoint(config, batch_size)
            t2i, i2t = metrics["text_to_image"], metrics["image_to_text"]
            val_losses = history.get("val_retrieval_loss") or history["val_loss"]
            row = {
                "experiment": name, "status": "completed",
                "batch_size": batch_size, "epochs_requested": epochs,
                "train_steps_per_epoch": train_steps_per_epoch,
                "validation_steps": validation_steps,
                "validation_loss_is_partial": True,
                "loss_type": config["loss_type"],
                "caption_count": config["caption_count"],
                "image_augmentation": config["image_augmentation"],
                "negative_sampling_strategy": config["negative_sampling_strategy"],
                "pretrained": config["pretrained"],
                "train_all_backbone": config["train_all_backbone"],
                "epochs_completed": len(history["train_loss"]),
                "best_val_retrieval_loss": min(val_losses),
                "final_val_retrieval_loss": val_losses[-1],
                "training_time_minutes": (time.time() - started) / 60.0,
                "overall_validation_mean_recall": (t2i["mean_recall"] + i2t["mean_recall"]) / 2.0,
                "text_to_image_r1": t2i["recall@1"],
                "text_to_image_r5": t2i["recall@5"],
                "text_to_image_r10": t2i["recall@10"],
                "image_to_text_r1": i2t["recall@1"],
                "image_to_text_r5": i2t["recall@5"],
                "image_to_text_r10": i2t["recall@10"],
                "text_to_image_mrr": t2i["mrr"],
                "image_to_text_mrr": i2t["mrr"],
                "checkpoint_path": metrics["checkpoint_path"],
                "evaluation_split": "validation", "test_set_used": False,
            }
            rows.append(row)
            print(f"[PASS] {name}: validation Mean Recall={row['overall_validation_mean_recall']:.4f}")
        except Exception as error:
            rows.append({
                "experiment": name, "status": "failed",
                "training_time_minutes": (time.time() - started) / 60.0,
                "error": str(error)[:1500],
            })
            print(f"[FAILED] {name}: {error}")
            traceback.print_exc()
        finally:
            tf.keras.backend.clear_session()
            save_results(rows)
    save_results(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 5 controlled data and initialization experiments.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--train-steps-per-epoch", type=int, default=250,
        help="Training batches per epoch (default: 250).",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=100,
        help="Validation batches per epoch; final retrieval evaluation still uses the full validation split.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device batch size. Default 4 is conservative for GPUs with 6 GB VRAM.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.train_steps_per_epoch <= 0:
        parser.error("--train-steps-per-epoch must be positive.")
    if args.validation_steps <= 0:
        parser.error("--validation-steps must be positive.")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined.")

    all_experiments = build_experiments()
    experiments = all_experiments
    if args.only:
        requested, known = set(args.only), {x["name"] for x in experiments}
        if requested - known:
            parser.error(f"Unknown experiments: {sorted(requested - known)}")
        experiments = [x for x in experiments if x["name"] in requested]

    print("PHASE 5 EXPERIMENT PLAN | ResNet50 + LSTM | Fold 1 | Seed 42")
    print("Validation uses all five captions. Test split is never loaded.")
    for x in experiments:
        print(
            f"{x['name']:<34} captions={x['caption_count']} "
            f"augmentation={x['image_augmentation']} loss={x['loss_type']} "
            f"negative={x['negative_sampling_strategy']} "
            f"pretrained={x['pretrained']} train_backbone={x['train_all_backbone']}"
        )
    if args.list:
        print("List-only mode. No training was started.")
        return
    run_all(
        experiments, all_experiments, args.epochs, args.batch_size,
        args.train_steps_per_epoch, args.validation_steps,
        args.resume, args.overwrite,
    )


if __name__ == "__main__":
    main()
