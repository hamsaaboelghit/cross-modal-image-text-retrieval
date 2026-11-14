
"""
Centralized model configurations for VisionMatch AI.

The vocabulary size is supplied at runtime.
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
    },

    "lstm": {
        "encoder_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
    },

    "gru": {
        "encoder_type": "gru",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
    },

    "bidirectional_rnn": {
        "encoder_type": "rnn",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
    },

    "bidirectional_lstm": {
        "encoder_type": "lstm",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
    },

    "bidirectional_gru": {
        "encoder_type": "gru",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": True,
        "use_attention": True,
    },

    "transformer": {
        "encoder_type": "transformer",
        "hidden_dim": 512,
        "num_layers": 1,
        "dropout": 0.2,
        "bidirectional": False,
        "use_attention": True,
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

# Shared settings
COMMON_CONFIG = {
    "embedding_dim": 300,
    "pretrained": True,
    "unfreeze_last_block": False,
    "projection_dropout": 0.1,
}
