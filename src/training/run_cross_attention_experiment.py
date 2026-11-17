from src.training.experiment_runner import run_experiment


def main():
    run_experiment(
        experiment_name="cross_attention_fold1_seed42",
        model_type="full",
        encoder_type="lstm",
        use_cross_modal_attention=True,
        pretrained=True,
        unfreeze_last_block=False,
        epochs=5,
        batch_size=8,
        seed=42,
        fold=1,
    )


if __name__ == "__main__":
    main()