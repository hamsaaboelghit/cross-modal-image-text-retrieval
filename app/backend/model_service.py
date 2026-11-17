
import json
import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.models.full_model import CrossModalRetrievalModel
from src.models.baseline_model import BaselineRetrievalModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_TYPE = os.getenv(
    "VISIONMATCH_MODEL_TYPE",
    "baseline",
).strip().lower()

CHECKPOINT_DIR = Path(
    os.getenv(
        "VISIONMATCH_CHECKPOINT_DIR",
        str(
            PROJECT_ROOT
            / "checkpoints"
            / "experiments"
            / "baseline_fold1_seed42"
        ),
    )
)

CHECKPOINT_KIND = os.getenv(
    "VISIONMATCH_CHECKPOINT_KIND",
    "best",
).strip().lower()

VOCAB_PATH = Path(
    os.getenv(
        "VISIONMATCH_VOCAB_PATH",
        str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "flickr30k"
            / "vocabularies"
            / "fold_1_vocab.json"
        ),
    )
)

IMAGE_SIZE = 224
MAX_LENGTH = 34
EMBEDDING_DIM = 512


class ModelService:
    """Load a trained VisionMatch model and generate embeddings."""

    def __init__(self):
        self.model = None
        self.vocabulary = None
        self.checkpoint_path = None
        self.checkpoint_info = {}
        self.model_type = MODEL_TYPE

    def _load_vocabulary(self):
        if not VOCAB_PATH.is_file():
            raise FileNotFoundError(
                f"Vocabulary not found: {VOCAB_PATH}"
            )

        with VOCAB_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            vocabulary = json.load(file)

        if not isinstance(vocabulary, dict):
            raise TypeError(
                "Vocabulary must be a dictionary."
            )

        required_tokens = {
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        }

        missing = required_tokens - vocabulary.keys()

        if missing:
            raise ValueError(
                f"Vocabulary is missing special tokens: {missing}"
            )

        token_ids = list(vocabulary.values())

        if not all(
            isinstance(token_id, int) and token_id >= 0
            for token_id in token_ids
        ):
            raise ValueError(
                "Vocabulary token IDs must be non-negative integers."
            )

        if len(set(token_ids)) != len(token_ids):
            raise ValueError(
                "Vocabulary contains duplicate token IDs."
            )

        self.vocabulary = vocabulary

    @staticmethod
    def _checkpoint_from_directory(directory):
        directory = Path(directory)

        if not directory.is_dir():
            return None

        return tf.train.latest_checkpoint(str(directory))

    def _resolve_checkpoint(self):
        path = CHECKPOINT_DIR.expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {path}"
            )

        # The environment variable may point directly to a checkpoint prefix.
        if path.is_file() and str(path).endswith(".index"):
            return str(path)[:-6]

        if path.is_file():
            return str(path)

        # Or it may point to an experiment root, latest/, or best/.
        if self._checkpoint_from_directory(path) is not None:
            return self._checkpoint_from_directory(path)

        if self.model_type == "baseline":
            preferred_kinds = [CHECKPOINT_KIND, "best", "latest"]
        else:
            preferred_kinds = [CHECKPOINT_KIND, "best", "latest"]

        checked = set()

        for kind in preferred_kinds:
            if kind in checked:
                continue

            checked.add(kind)

            if kind not in {"best", "latest"}:
                continue

            candidate_dir = path / kind
            checkpoint = self._checkpoint_from_directory(
                candidate_dir
            )

            if checkpoint is not None:
                return checkpoint

        raise FileNotFoundError(
            "No TensorFlow checkpoint was found.\n"
            f"Checked: {path}\n"
            f"Also checked: {path / 'best'} and {path / 'latest'}\n"
            "Set VISIONMATCH_CHECKPOINT_DIR to the experiment "
            "directory, a checkpoint directory, or a checkpoint prefix."
        )

    def _build_model(self):
        vocab_size = len(self.vocabulary)

        if self.model_type == "baseline":
            model = BaselineRetrievalModel(
                vocab_size=vocab_size,
                embedding_dim=300,
                output_dim=EMBEDDING_DIM,
                dropout=0.1,
            )

        elif self.model_type in {
            "rnn",
            "lstm",
            "gru",
            "transformer",
            "cross_attention",
        }:
            use_cross_attention = (
                self.model_type == "cross_attention"
            )

            encoder_type = (
                "lstm"
                if use_cross_attention
                else self.model_type
            )

            model = CrossModalRetrievalModel(
                vocab_size=vocab_size,
                encoder_type=encoder_type,
                embedding_dim=300,
                hidden_dim=512,
                num_layers=1,
                dropout=0.2,
                bidirectional=False,
                use_attention=True,
                pretrained=False,
                unfreeze_last_block=False,
                use_cross_modal_attention=use_cross_attention,
            )

        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}"
            )

        # Build model variables before restoring checkpoint weights.
        dummy_images = tf.zeros(
            (1, IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=tf.float32,
        )

        dummy_captions = tf.zeros(
            (1, 5, MAX_LENGTH),
            dtype=tf.int32,
        )

        model(
            dummy_images,
            dummy_captions,
            training=False,
        )

        return model

    def load(self):
        if self.is_loaded:
            return self

        print("Loading vocabulary...")
        self._load_vocabulary()

        print("Building TensorFlow model...")
        self.model = self._build_model()

        print("Resolving TensorFlow checkpoint...")
        self.checkpoint_path = self._resolve_checkpoint()

        print("Restoring model weights...")

        checkpoint = tf.train.Checkpoint(
            model=self.model,
        )

        status = checkpoint.restore(
            self.checkpoint_path
        )

        # Training checkpoints can also contain optimizer and state variables.
        status.expect_partial()

        self.checkpoint_info = {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_kind": CHECKPOINT_KIND,
            "vocab_size": len(self.vocabulary),
            "embedding_dim": EMBEDDING_DIM,
        }

        print("TensorFlow model loaded successfully.")
        print("Model type:", self.model_type)
        print("Checkpoint:", self.checkpoint_path)

        return self

    def _tokenize(self, caption):
        if (
            not isinstance(caption, str)
            or not caption.strip()
        ):
            raise ValueError(
                "Caption must be a non-empty string."
            )

        tokens = re.findall(
            r"\w+|[^\w\s]",
            caption.lower(),
        )

        encoded = [
            self.vocabulary["<BOS>"],
        ]

        encoded.extend(
            self.vocabulary.get(
                token,
                self.vocabulary["<UNK>"],
            )
            for token in tokens
        )

        encoded.append(
            self.vocabulary["<EOS>"]
        )

        if len(encoded) > MAX_LENGTH:
            encoded = encoded[:MAX_LENGTH]
            encoded[-1] = self.vocabulary["<EOS>"]

        padding_id = self.vocabulary["<PAD>"]

        encoded.extend(
            [padding_id] * (MAX_LENGTH - len(encoded))
        )

        return tf.convert_to_tensor(
            [encoded],
            dtype=tf.int32,
        )

    def encode_caption(self, caption):
        if not self.is_loaded:
            raise RuntimeError(
                "Model service is not loaded."
            )

        return self._tokenize(caption)

    def get_text_embedding(self, caption):
        if not self.is_loaded:
            raise RuntimeError(
                "Model service is not loaded."
            )

        tokens = self._tokenize(caption)

        if self.model_type == "baseline":
            embedding = self.model.text_encoder(
                tokens,
                training=False,
            )
        else:
            embedding = self.model.encode_text(
                tokens,
                training=False,
            )

        return tf.math.l2_normalize(
            embedding,
            axis=-1,
        )

    def preprocess_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Expected a PIL Image."
            )

        image = image.convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        return tf.convert_to_tensor(
            image_array[np.newaxis, ...],
            dtype=tf.float32,
        )

    def get_image_embedding(self, image):
        if not self.is_loaded:
            raise RuntimeError(
                "Model service is not loaded."
            )

        image_tensor = self.preprocess_image(image)

        if self.model_type == "baseline":
            embedding = self.model.image_encoder(
                image_tensor,
                training=False,
            )
        else:
            embedding = self.model.encode_image(
                image_tensor,
                training=False,
            )

        return tf.math.l2_normalize(
            embedding,
            axis=-1,
        )

    @property
    def is_loaded(self):
        return (
            self.model is not None
            and self.vocabulary is not None
        )


if __name__ == "__main__":
    service = ModelService()
    service.load()

    print("\nModel service is ready.")
    print("Loaded model:", service.model_type)
    print("Checkpoint:", service.checkpoint_path)
