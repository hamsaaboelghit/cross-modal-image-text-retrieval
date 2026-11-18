
from src.evaluation import evaluation_runner


def main():
    evaluation_runner.EXPERIMENTS["lstm"] = {
        "model_type": "full",
        "model_name": "ResNet50-LSTM",
        "encoder_type": "lstm",
        "use_cross_modal_attention": False,
        "pretrained": True,
        "unfreeze_last_block": False,
        "embedding_dim": 300,
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "checkpoint_dir": (
            "checkpoints/experiments/lstm_fold1_seed42"
        ),
        "output_dir": (
            "results/experiments/lstm_fold1_seed42"
        ),
    }

    evaluation_runner.run_evaluation(
        experiment_name="lstm",
        checkpoint_kind="best",
    )


if __name__ == "__main__":
    main()
