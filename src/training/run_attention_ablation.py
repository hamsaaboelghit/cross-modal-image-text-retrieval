from src.training.experiment_runner import run_experiment

common = dict(
    model_type="full",
    encoder_type="gru",
    pretrained=True,
    unfreeze_last_block=False,
    epochs=1,
    batch_size=4,
    seed=42,
    fold=1,
    learning_rate=0.0002,
    temperature=0.07,
    dropout=0.1,
    max_train_steps_per_epoch=200,
    max_validation_steps=100,
    log_dir="runs/attention_ablation",
)

print("\n========== ABLATION A: WITHOUT CROSS-MODAL ATTENTION ==========")
run_experiment(
    experiment_name="ablation_gru_attention_off",
    use_cross_modal_attention=False,
    **common,
)

print("\n========== ABLATION B: WITH CROSS-MODAL ATTENTION ==========")
run_experiment(
    experiment_name="ablation_gru_attention_on",
    use_cross_modal_attention=True,
    **common,
)

print("\n========== ABLATION STUDY FINISHED ==========")
