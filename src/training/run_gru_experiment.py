from src.training.experiment_runner import run_experiment


def main():
    run_experiment(
        experiment_name="gru_fold1_seed42",
        model_type="full",
        encoder_type="gru",
        use_cross_modal_attention=False,
        pretrained=True,
        unfreeze_last_block=False,
        epochs=5,
        batch_size=8,
        dropout=0.1,
        seed=42,
        fold=1,
    )


if __name__ == "__main__":
    main()