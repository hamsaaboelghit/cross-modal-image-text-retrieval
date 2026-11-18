
from experiments.model_configs import (
    BASELINE_CONFIG,
    MODEL_CONFIGS,
    COMMON_CONFIG,
    MATCHING_EXPERIMENTS,
    get_model_config,
    get_matching_experiment,
    get_matching_training_config,
)

from src.models import (
    BaselineRetrievalModel,
    CrossModalRetrievalModel,
)


def test_model_configs():

    expected_models = {
        "rnn",
        "lstm",
        "gru",
        "bidirectional_rnn",
        "bidirectional_lstm",
        "bidirectional_gru",
        "transformer",
        "advanced_cross_modal",
    }

    assert set(MODEL_CONFIGS.keys()) == expected_models

    for name, config in MODEL_CONFIGS.items():

        assert "encoder_type" in config
        assert "hidden_dim" in config
        assert "use_attention" in config
        assert "use_cross_modal_attention" in config

        model_config = get_model_config(
            name,
            pretrained=False,
        )

        model = CrossModalRetrievalModel(
            vocab_size=1000,
            **model_config,
        )

        assert model.embedding_dim == 512

        if name == "advanced_cross_modal":
            assert model.use_cross_modal_attention is True
        else:
            assert model.use_cross_modal_attention is False

        del model

        print(f"{name}: PASS")

    # Baseline model
    baseline = BaselineRetrievalModel(
        vocab_size=1000,
        **BASELINE_CONFIG,
    )

    assert baseline.embedding_dim == 512

    print("baseline: PASS")

    # Shared configuration
    assert COMMON_CONFIG["embedding_dim"] == 300
    assert COMMON_CONFIG["projection_dropout"] == 0.1

    print("Shared model configuration: PASS")

    print("All model configuration tests passed!")


def test_matching_experiment_configs():

    expected_experiments = {
        "bi_encoder_baseline",
        "matching_low",
        "matching_default",
        "matching_high",
    }

    assert set(MATCHING_EXPERIMENTS.keys()) == expected_experiments

    expected_weights = {
        "bi_encoder_baseline": 0.0,
        "matching_low": 0.1,
        "matching_default": 0.2,
        "matching_high": 0.5,
    }

    for name, expected_weight in expected_weights.items():

        experiment = get_matching_experiment(name)

        assert (
            experiment["matching_loss_weight"]
            == expected_weight
        )

        training_config = get_matching_training_config(name)

        assert (
            training_config["matching_loss_weight"]
            == expected_weight
        )

        if name == "bi_encoder_baseline":
            assert experiment["use_cross_modal_attention"] is False
        else:
            assert experiment["use_cross_modal_attention"] is True

        print(f"{name}: PASS")

    # Invalid experiment name
    try:
        get_matching_experiment("unknown_experiment")
        raise AssertionError(
            "Unknown matching experiment was accepted."
        )
    except ValueError:
        print("Invalid experiment name validation: PASS")

    # Invalid model name
    try:
        get_model_config("unknown_model")
        raise AssertionError(
            "Unknown model configuration was accepted."
        )
    except ValueError:
        print("Invalid model name validation: PASS")

    print("All matching experiment tests passed!")


if __name__ == "__main__":
    test_model_configs()
    test_matching_experiment_configs()
