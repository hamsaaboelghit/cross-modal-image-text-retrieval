"""Continue the five fast architecture runs from 1,000 to 1,500 batches.

The original checkpoints and logs are read-only inputs. All resumed outputs are
written to new experiment directories and a new log directory.
"""
import json
import time
from pathlib import Path

import tensorflow as tf

from src.training.experiment_runner import run_experiment


ORIGINAL_PREFIX = "deadline_fast_20261119"
RESUME_PREFIX = "deadline_resume500_20261120"

BATCH_SIZE = 4
SEED = 42
FOLD = 1
LEARNING_RATE = 1e-4
OUTPUT_DIM = 512
DROPOUT = 0.1

PREVIOUS_TRAIN_BATCHES = 1000
ADDITIONAL_TRAIN_BATCHES = 500
TOTAL_TRAIN_BATCHES = PREVIOUS_TRAIN_BATCHES + ADDITIONAL_TRAIN_BATCHES
VALIDATION_STEPS = 200

EXPERIMENTS = [
    {"name": "rnn", "encoder_type": "rnn", "cross_attention": False},
    {"name": "lstm", "encoder_type": "lstm", "cross_attention": False},
    {"name": "gru", "encoder_type": "gru", "cross_attention": False},
    {"name": "transformer", "encoder_type": "transformer", "cross_attention": False},
    {"name": "cross_attention", "encoder_type": "lstm", "cross_attention": True},
]


def validate_source_checkpoint(model_name):
    """Refuse to resume unless the original 1,000-step checkpoint is present."""
    source_dir = (
        Path("checkpoints")
        / "experiments"
        / f"{ORIGINAL_PREFIX}_{model_name}_fold1_seed42"
    )
    latest_dir = source_dir / "latest"

    if not latest_dir.is_dir():
        raise FileNotFoundError(
            f"Original latest checkpoint directory is missing: {latest_dir}"
        )

    checkpoint_prefix = tf.train.latest_checkpoint(str(latest_dir))
    if checkpoint_prefix is None:
        raise FileNotFoundError(
            f"No TensorFlow latest checkpoint found in: {latest_dir}"
        )

    metadata_path = Path(checkpoint_prefix + ".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint metadata is missing: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    actual_step = int(metadata.get("global_step", -1))
    actual_epoch = int(metadata.get("epoch", -1))

    if actual_step != PREVIOUS_TRAIN_BATCHES or actual_epoch != 0:
        raise RuntimeError(
            f"{model_name}: expected original checkpoint at epoch=0 and "
            f"global_step={PREVIOUS_TRAIN_BATCHES}, but found "
            f"epoch={actual_epoch}, global_step={actual_step}. "
            "Nothing was resumed for this model."
        )

    print(f"Verified source checkpoint for {model_name}:")
    print(f"  {checkpoint_prefix}")
    print(f"  epoch={actual_epoch}, global_step={actual_step}")

    # Pass the experiment directory: CheckpointManager resolves its /latest child.
    return str(source_dir)


def main():
    print("=" * 72)
    print("VISIONMATCH AI - RESUME FIVE MODELS FROM 1000 TO 1500 BATCHES")
    print("=" * 72)
    print("Original checkpoints/logs: PRESERVED (read-only)")
    print(f"Previously completed batches: {PREVIOUS_TRAIN_BATCHES}")
    print(f"Additional batches per model: {ADDITIONAL_TRAIN_BATCHES}")
    print(f"Target cumulative batches: {TOTAL_TRAIN_BATCHES}")
    print("Baseline: SKIPPED (full training already completed)")
    print("Execution: one model at a time; no parallel training")
    print("=" * 72)

    results = []

    for index, experiment in enumerate(EXPERIMENTS, start=1):
        name = experiment["name"]
        print("\n" + "=" * 72)
        print(f"MODEL {index}/{len(EXPERIMENTS)}: {name.upper()}")
        print("=" * 72)

        # Validate the original checkpoint before creating/running this continuation.
        source_checkpoint = validate_source_checkpoint(name)
        resumed_name = f"{RESUME_PREFIX}_{name}_fold1_seed42"

        start_time = time.time()
        try:
            history = run_experiment(
                experiment_name=resumed_name,
                model_type="full",
                encoder_type=experiment["encoder_type"],
                use_cross_modal_attention=experiment["cross_attention"],
                pretrained=True,
                unfreeze_last_block=False,
                train_all_backbone=False,
                caption_count=5,
                image_augmentation="none",
                negative_sampling_strategy="random",
                epochs=2,  # checkpoint is epoch 0; this runs the next epoch only
                batch_size=BATCH_SIZE,
                seed=SEED,
                fold=FOLD,
                learning_rate=LEARNING_RATE,
                temperature=0.07,
                loss_type="info_nce",
                margin=None,
                output_dim=OUTPUT_DIM,
                dropout=DROPOUT,
                log_dir="runs/deadline_resume500_20261120",
                max_train_steps_per_epoch=ADDITIONAL_TRAIN_BATCHES,
                max_validation_steps=VALIDATION_STEPS,
                resume_from=source_checkpoint,
                resume_skip_batches=PREVIOUS_TRAIN_BATCHES,
            )

            elapsed = (time.time() - start_time) / 60.0
            val_losses = history.get("val_retrieval_loss") or history.get("val_loss") or []
            result = {
                "model": name,
                "status": "COMPLETED",
                "cumulative_train_batches": TOTAL_TRAIN_BATCHES,
                "new_validation_loss": min(val_losses) if val_losses else None,
                "training_time_minutes": round(elapsed, 2),
                "new_experiment": resumed_name,
            }
            results.append(result)
            print(f"\n[COMPLETED] {name.upper()}: {result}")

        except Exception as error:
            elapsed = (time.time() - start_time) / 60.0
            print(f"\n[FAILED] {name.upper()}: {error}")
            print("Stopping here to avoid silently continuing with an uncertain state.")
            raise

        finally:
            tf.keras.backend.clear_session()

    print("\n" + "=" * 72)
    print("RESUME SUMMARY")
    print("=" * 72)
    for result in results:
        print(result)
    print("\nAll five models reached the requested continuation stage.")


if __name__ == "__main__":
    main()
