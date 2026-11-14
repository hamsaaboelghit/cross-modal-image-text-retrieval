
from experiments.model_configs import (
    BASELINE_CONFIG,
    MODEL_CONFIGS,
    COMMON_CONFIG,
)

from src.models import BaselineRetrievalModel, CrossModalRetrievalModel


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

        model_config = {
            **COMMON_CONFIG,
            **config,
        }

        model_config["pretrained"] = False

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

    baseline = BaselineRetrievalModel(
        vocab_size=1000,
        **BASELINE_CONFIG,
    )

    assert baseline.embedding_dim == 512

    print("baseline: PASS")
    print("All model configuration tests passed!")


if __name__ == "__main__":
    test_model_configs()
