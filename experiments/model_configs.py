
"""
Centralized model and experiment configurations for VisionMatch AI.

The vocabulary size is supplied at runtime.

Model architecture settings are kept separate from training settings.
"""

# 1. Baseline Model
BASELINE_CONFIG = {
    "embedding_dim": 300,
    "output_dim": 512,
    "dropout": 0.1,
}


# 2. Classic and Modern Architectures
MODEL_CONFIGS = {
    "rnn": {
        "encoder_type": "rnn",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    "lstm": {
        "encoder_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    "gru": {
        "encoder_type": "gru",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    "bidirectional_rnn": {
        "encoder_type": "rnn",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    "bidirectional_lstm": {
        "encoder_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    "bidirectional_gru": {
        "encoder_type": "gru",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    "transformer": {
        "encoder_type": "transformer",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "use_cross_modal_attention": False,
    },

    # 3. Advanced Cross-Modal Architecture
    "advanced_cross_modal": {
        "encoder_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
        "use_cross_modal_attention": True,
        "cross_modal_num_heads": 8,
        "cross_modal_dropout": 0.1,
    },
}


# 4. Shared Model Settings
COMMON_CONFIG = {
    "embedding_dim": 300,
    "pretrained": True,
    "unfreeze_last_block": False,
    "projection_dropout": 0.1,
}


# 5. Matching Loss Experiments
#
# These weights are passed to TrainingConfig, not to the model.
# A weight of 0.0 disables the matching-loss contribution.
MATCHING_EXPERIMENTS = {
    "bi_encoder_baseline": {
        "use_cross_modal_attention": False,
        "matching_loss_weight": 0.0,
    },

    "matching_low": {
        "use_cross_modal_attention": True,
        "matching_loss_weight": 0.1,
    },

    "matching_default": {
        "use_cross_modal_attention": True,
        "matching_loss_weight": 0.2,
    },

    "matching_high": {
        "use_cross_modal_attention": True,
        "matching_loss_weight": 0.5,
    },
}


def get_model_config(model_name, pretrained=None):
    """
    Return a new model configuration dictionary.

    Args:
        model_name: Key from MODEL_CONFIGS.
        pretrained: Optional override for pretrained image weights.
    """

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model configuration: {model_name}. "
            f"Available models: {list(MODEL_CONFIGS)}"
        )

    config = {
        **COMMON_CONFIG,
        **MODEL_CONFIGS[model_name],
    }

    if pretrained is not None:
        config["pretrained"] = bool(pretrained)

    return config


def get_matching_experiment(experiment_name):
    """
    Return a copy of a matching experiment configuration.
    """

    if experiment_name not in MATCHING_EXPERIMENTS:
        raise ValueError(
            f"Unknown matching experiment: {experiment_name}. "
            f"Available experiments: {list(MATCHING_EXPERIMENTS)}"
        )

    return dict(MATCHING_EXPERIMENTS[experiment_name])


def get_matching_training_config(experiment_name):
    """
    Return the TrainingConfig arguments for a matching experiment.

    The caller can use:
        TrainingConfig(**get_matching_training_config(name))
    """

    experiment = get_matching_experiment(experiment_name)

    return {
        "matching_loss_weight": experiment["matching_loss_weight"],
    }
