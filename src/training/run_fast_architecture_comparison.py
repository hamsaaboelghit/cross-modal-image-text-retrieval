import time
import tensorflow as tf

from src.training.experiment_runner import run_experiment


# ==========================================
# FAST ARCHITECTURE COMPARISON
# ==========================================

EPOCHS = 1
BATCH_SIZE = 4
TRAIN_STEPS = 1000
VALIDATION_STEPS = 200

SEED = 42
FOLD = 1
LEARNING_RATE = 1e-4
OUTPUT_DIM = 512
DROPOUT = 0.1

RUN_PREFIX = "deadline_fast_20261119"

# Baseline is intentionally excluded because
# its full training has already been completed.

EXPERIMENTS = [
    {
        "name": "rnn",
        "encoder_type": "rnn",
        "cross_attention": False,
    },
    {
        "name": "lstm",
        "encoder_type": "lstm",
        "cross_attention": False,
    },
    {
        "name": "gru",
        "encoder_type": "gru",
        "cross_attention": False,
    },
    {
        "name": "transformer",
        "encoder_type": "transformer",
        "cross_attention": False,
    },
    {
        "name": "cross_attention",
        "encoder_type": "lstm",
        "cross_attention": True,
    },
]


def main():

    print("=" * 65)
    print("VISIONMATCH AI - FAST ARCHITECTURE COMPARISON")
    print("=" * 65)

    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training batches per epoch: {TRAIN_STEPS}")
    print(f"Validation batches: {VALIDATION_STEPS}")
    print("Baseline: SKIPPED (already completed)")
    print("=" * 65)

    results = []

    for index, experiment in enumerate(EXPERIMENTS, start=1):

        name = experiment["name"]

        experiment_name = f"{RUN_PREFIX}_{name}_fold1_seed42"

        print("\n" + "=" * 65)
        print(f"MODEL {index}/{len(EXPERIMENTS)}: {name.upper()}")
        print("=" * 65)

        start_time = time.time()

        try:

            history = run_experiment(
                experiment_name=experiment_name,

                model_type="full",
                encoder_type=experiment["encoder_type"],

                use_cross_modal_attention=experiment["cross_attention"],

                pretrained=True,
                unfreeze_last_block=False,
                train_all_backbone=False,

                caption_count=5,
                image_augmentation="none",
                negative_sampling_strategy="random",

                epochs=EPOCHS,
                batch_size=BATCH_SIZE,

                seed=SEED,
                fold=FOLD,

                learning_rate=LEARNING_RATE,
                temperature=0.07,

                loss_type="info_nce",
                margin=None,

                output_dim=OUTPUT_DIM,
                dropout=DROPOUT,

                log_dir="runs/deadline_fast_20261119",

                max_train_steps_per_epoch=TRAIN_STEPS,
                max_validation_steps=VALIDATION_STEPS,
            )

            elapsed = (time.time() - start_time) / 60

            val_losses = (
                history.get("val_retrieval_loss")
                or history.get("val_loss")
                or []
            )

            best_val = min(val_losses) if val_losses else None

            results.append({
                "model": name,
                "status": "COMPLETED",
                "best_validation_loss": best_val,
                "training_time_minutes": round(elapsed, 2),
                "experiment_name": experiment_name,
            })

            print(f"\n[COMPLETED] {name.upper()}")
            print(f"Time: {elapsed:.2f} minutes")

            if best_val is not None:
                print(f"Best validation loss: {best_val:.6f}")

        except Exception as error:

            elapsed = (time.time() - start_time) / 60

            results.append({
                "model": name,
                "status": "FAILED",
                "error": str(error),
                "training_time_minutes": round(elapsed, 2),
                "experiment_name": experiment_name,
            })

            print(f"\n[FAILED] {name.upper()}: {error}")

        finally:
            tf.keras.backend.clear_session()

    print("\n" + "=" * 65)
    print("FAST COMPARISON SUMMARY")
    print("=" * 65)

    for result in results:
        print(result)

    print("\nAll requested experiments have been processed.")


if __name__ == "__main__":
    main()
