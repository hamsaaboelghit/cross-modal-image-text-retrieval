
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

MODEL_TYPE = os.getenv("VISIONMATCH_MODEL_TYPE", "baseline").lower()

CHECKPOINT_DIR = Path(
    os.getenv(
        "VISIONMATCH_CHECKPOINT_DIR",
        str(
            PROJECT_ROOT
            / "checkpoints"
            / "experiments"
            / "baseline_fold1"
        ),
    )
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / "fold_1_vocab.json"
)

IMAGE_SIZE = 224
MAX_LENGTH = 34
EMBEDDING_DIM = 512


class ModelService:
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

        with VOCAB_PATH.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)

        if not isinstance(vocabulary, dict):
            raise TypeError("Vocabulary must be a dictionary.")

        required_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        missing = required_tokens - vocabulary.keys()

        if missing:
            raise ValueError(
                f"Vocabulary is missing special tokens: {missing}"
            )

        self.vocabulary = vocabulary

    def _resolve_checkpoint(self):
        if not CHECKPOINT_DIR.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {CHECKPOINT_DIR}"
            )

        checkpoint_path = tf.train.latest_checkpoint(
            str(CHECKPOINT_DIR)
        )

        if checkpoint_path is None:
            raise FileNotFoundError(
                "No TensorFlow checkpoint was found in "
                f"{CHECKPOINT_DIR}. Make sure training has saved "
                "at least one checkpoint."
            )

        return checkpoint_path

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

        # Build all model variables before restoring the checkpoint.
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
        checkpoint = tf.train.Checkpoint(model=self.model)

        status = checkpoint.restore(self.checkpoint_path)

        # The training checkpoint may also contain optimizer and
        # training-state variables. This service only needs the model.
        status.expect_partial()

        self.checkpoint_info = {
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "vocab_size": len(self.vocabulary),
            "embedding_dim": EMBEDDING_DIM,
        }

        print("TensorFlow model loaded successfully.")
        print("Model type:", self.model_type)
        print("Checkpoint:", self.checkpoint_path)

        return self

    def _tokenize(self, caption):
        if not isinstance(caption, str) or not caption.strip():
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

        encoded.append(self.vocabulary["<EOS>"])

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
            raise RuntimeError("Model service is not loaded.")

        return self._tokenize(caption)

    def get_text_embedding(self, caption):
        if not self.is_loaded:
            raise RuntimeError("Model service is not loaded.")

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
            raise TypeError("Expected a PIL Image.")

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
            raise RuntimeError("Model service is not loaded.")

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
