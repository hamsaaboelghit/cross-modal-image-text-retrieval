from src.training.experiment_runner import run_experiment


def main():
    run_experiment(
        experiment_name="baseline_fold1_seed42",
        model_type="baseline",
        epochs=5,
        batch_size=8,
        seed=42,
        fold=1,
    )


if __name__ == "__main__":
    main()