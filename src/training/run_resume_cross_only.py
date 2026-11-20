import gc
import json
import time
from pathlib import Path

import tensorflow as tf

from src.training.experiment_runner import run_experiment

SOURCE_NAME = "deadline_fast_20261119_cross_attention_fold1_seed42"
RESUME_NAME = "deadline_resume500_20261120_cross_attention_fold1_seed42"

SOURCE_DIR = Path("checkpoints/experiments") / SOURCE_NAME
OUTPUT_DIR = Path("checkpoints/experiments") / RESUME_NAME

PREVIOUS_BATCHES = 1000
ADDITIONAL_BATCHES = 500

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_DIR}\n"
        "Stopping to protect previous checkpoints."
    )

latest_dir = SOURCE_DIR / "latest"
checkpoint = tf.train.latest_checkpoint(str(latest_dir))

if checkpoint is None:
    raise FileNotFoundError(f"No checkpoint found in {latest_dir}")

metadata_path = Path(checkpoint + ".json")
if not metadata_path.exists():
    raise FileNotFoundError(f"Missing metadata: {metadata_path}")

with metadata_path.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

if int(metadata.get("global_step", -1)) != PREVIOUS_BATCHES:
    raise RuntimeError(
        f"Expected step 1000, found {metadata.get('global_step')}"
    )

print("=" * 65)
print("CROSS-ATTENTION: RESUME FROM 1000 TO 1500 BATCHES")
print(f"Source: {checkpoint}")
print(f"Additional batches: {ADDITIONAL_BATCHES}")
print("Execution: ONE MODEL ONLY")
print("Test dataset: NOT USED")
print("=" * 65)

start = time.time()

try:
    history = run_experiment(
        experiment_name=RESUME_NAME,
        model_type="full",
        encoder_type="lstm",
        use_cross_modal_attention=True,
        pretrained=True,
        unfreeze_last_block=False,
        train_all_backbone=False,
        caption_count=5,
        image_augmentation="none",
        negative_sampling_strategy="random",
        epochs=2,
        batch_size=4,
        seed=42,
        fold=1,
        learning_rate=1e-4,
        temperature=0.07,
        loss_type="info_nce",
        margin=None,
        output_dim=512,
        dropout=0.1,
        log_dir="runs/deadline_resume500_20261120",
        max_train_steps_per_epoch=ADDITIONAL_BATCHES,
        max_validation_steps=200,
        resume_from=str(SOURCE_DIR),
        resume_skip_batches=PREVIOUS_BATCHES,
    )

    elapsed = (time.time() - start) / 60
    val_losses = (
        history.get("val_retrieval_loss")
        or history.get("val_loss")
        or []
    )

    print("\nCROSS-ATTENTION COMPLETED")
    print(f"Training time: {elapsed:.2f} minutes")
    print(f"Validation retrieval loss: {min(val_losses) if val_losses else 'N/A'}")
    print(f"Checkpoint: {OUTPUT_DIR}")

finally:
    tf.keras.backend.clear_session()
    gc.collect()
