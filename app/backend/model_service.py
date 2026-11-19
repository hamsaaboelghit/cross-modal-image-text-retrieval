
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv

from src.data.vocabulary import load_vocabulary
from src.models.full_model import CrossModalRetrievalModel
from src.models.baseline_model import BaselineRetrievalModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

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
).expanduser()
if not VOCAB_PATH.is_absolute():
    VOCAB_PATH = PROJECT_ROOT / VOCAB_PATH

IMAGE_SIZE = 224
MAX_LENGTH = 34
NUM_CAPTIONS = 5
EMBEDDING_DIM = 512


def sha256_file(path):
    """Hash a file incrementally so large checkpoints do not fill memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint_prefix):
    """Fingerprint the index and every data shard of a TensorFlow checkpoint."""
    prefix = Path(checkpoint_prefix)
    files = [Path(str(prefix) + ".index")]
    files.extend(sorted(prefix.parent.glob(prefix.name + ".data-*")))
    if len(files) < 2 or any(not item.is_file() for item in files):
        raise FileNotFoundError(
            f"Cannot fingerprint incomplete TensorFlow checkpoint: {checkpoint_prefix}"
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


class ModelService:
    """Load a trained VisionMatch model and generate embeddings."""

    def __init__(self):
        self.model = None
        self.vocabulary = None
        self.checkpoint_path = None
        self.checkpoint_info = {}
        self.model_type = MODEL_TYPE

    # --------------------------------------------------
    # Vocabulary
    # --------------------------------------------------

    def _load_vocabulary(self):
        self.vocabulary = load_vocabulary(VOCAB_PATH)

        print(
            "Vocabulary loaded. Size:",
            len(self.vocabulary),
        )

    # --------------------------------------------------
    # Checkpoint resolution
    # --------------------------------------------------

    @staticmethod
    def _is_valid_checkpoint_prefix(prefix):
        prefix = Path(prefix)

        index_file = Path(str(prefix) + ".index")

        data_files = list(
            prefix.parent.glob(
                prefix.name + ".data-*"
            )
        )

        return index_file.is_file() and bool(data_files)

    @classmethod
    def _checkpoint_from_directory(cls, directory):
        directory = Path(directory)

        if not directory.is_dir():
            return None

        checkpoint = tf.train.latest_checkpoint(
            str(directory)
        )

        if (
            checkpoint is not None
            and cls._is_valid_checkpoint_prefix(checkpoint)
        ):
            return checkpoint

        # Fallback for directories where checkpoint state
        # metadata is missing but checkpoint files exist.
        index_files = list(
            directory.glob("*.index")
        )

        if not index_files:
            return None

        index_files.sort(
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

        for index_file in index_files:
            prefix = str(index_file)[:-6]

            if cls._is_valid_checkpoint_prefix(prefix):
                return prefix

        return None

    def _resolve_checkpoint(self):
        if CHECKPOINT_KIND not in {"best", "latest"}:
            raise ValueError(
                "VISIONMATCH_CHECKPOINT_KIND must be "
                "'best' or 'latest'."
            )

        path = CHECKPOINT_DIR.expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        # Case 1: A direct .index file was supplied.
        if path.suffix == ".index":
            prefix = str(path)[:-6]

            if self._is_valid_checkpoint_prefix(prefix):
                return prefix

            raise FileNotFoundError(
                f"Incomplete TensorFlow checkpoint: {path}"
            )

        # Case 2: A direct checkpoint prefix was supplied.
        if self._is_valid_checkpoint_prefix(path):
            return str(path)

        # Case 3: A directory was supplied.
        if not path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint path does not exist "
                f"or is not a valid checkpoint: {path}"
            )

        # First, check whether this directory itself contains
        # a TensorFlow checkpoint.
        direct_checkpoint = (
            self._checkpoint_from_directory(path)
        )

        if direct_checkpoint is not None:
            return direct_checkpoint

        # Otherwise, search the requested subdirectory first,
        # then fall back to the other checkpoint type.
        fallback_kind = (
            "latest"
            if CHECKPOINT_KIND == "best"
            else "best"
        )

        for kind in (CHECKPOINT_KIND, fallback_kind):
            candidate_dir = path / kind

            checkpoint = (
                self._checkpoint_from_directory(
                    candidate_dir
                )
            )

            if checkpoint is not None:
                return checkpoint

        raise FileNotFoundError(
            "No complete TensorFlow checkpoint was found.\n"
            f"Experiment directory: {path}\n"
            f"Checked: {path / CHECKPOINT_KIND}\n"
            f"Checked: {path / fallback_kind}\n"
            "Make sure both .index and .data files exist."
        )

    # --------------------------------------------------
    # Model construction
    # --------------------------------------------------

    def _build_model(self):
        if self.vocabulary is None:
            raise RuntimeError(
                "Load the vocabulary before building the model."
            )

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

        # The text encoders reject captions containing only PAD.
        # Build valid dummy captions using the real vocabulary IDs.
        pad_id = self.vocabulary["<PAD>"]
        bos_id = self.vocabulary["<BOS>"]
        eos_id = self.vocabulary["<EOS>"]

        valid_caption = (
            [bos_id, eos_id]
            + [pad_id] * (MAX_LENGTH - 2)
        )

        dummy_captions = tf.constant(
            [[valid_caption] * NUM_CAPTIONS],
            dtype=tf.int32,
        )

        dummy_images = tf.zeros(
            (1, IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=tf.float32,
        )

        # Build all model variables before restoring weights.
        model(
            dummy_images,
            dummy_captions,
            training=False,
        )

        return model

    # --------------------------------------------------
    # Loading and restoring
    # --------------------------------------------------

    def load(self):
        if self.is_loaded:
            return self

        print("Loading vocabulary...")
        self._load_vocabulary()

        print("Building TensorFlow model...")
        model = self._build_model()

        print("Resolving TensorFlow checkpoint...")
        checkpoint_path = self._resolve_checkpoint()

        print("Restoring model weights...")
        checkpoint = tf.train.Checkpoint(
            model=model,
        )

        status = checkpoint.restore(
            checkpoint_path
        )

        # Ensure that at least one checkpoint object matched,
        # and that all existing model variables were restored.
        status.assert_nontrivial_match()
        status.assert_existing_objects_matched()

        # The training checkpoint also contains optimizer and
        # training-state variables that are not needed for inference.
        status.expect_partial()

        # Mark the service as loaded only after successful restore.
        self.model = model
        self.checkpoint_path = checkpoint_path

        checkpoint_kind = Path(
            checkpoint_path
        ).parent.name

        self.checkpoint_info = {
            "model_type": self.model_type,
            "checkpoint_path": checkpoint_path,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint_path),
            "vocab_fingerprint": sha256_file(VOCAB_PATH),
            "vocab_size": len(self.vocabulary),
            "embedding_dim": EMBEDDING_DIM,
        }

        print("TensorFlow model loaded successfully.")
        print("Model type:", self.model_type)
        print("Checkpoint kind:", checkpoint_kind)
        print("Checkpoint:", checkpoint_path)

        return self

    # --------------------------------------------------
    # Text preprocessing
    # --------------------------------------------------

    def _tokenize(self, caption):
        if (
            not isinstance(caption, str)
            or not caption.strip()
        ):
            raise ValueError(
                "Caption must be a non-empty string."
            )

        # Keep exactly the same tokenization rule used by
        # build_vocabulary.py and encode_captions.py.
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

        # Match encode_captions.py truncation:
        # retain MAX_LENGTH - 1 tokens, then force EOS.
        if len(encoded) > MAX_LENGTH:
            encoded = encoded[:MAX_LENGTH - 1]
            encoded.append(
                self.vocabulary["<EOS>"]
            )

        padding_id = self.vocabulary["<PAD>"]

        encoded.extend(
            [padding_id] * (
                MAX_LENGTH - len(encoded)
            )
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

    # --------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------

    def preprocess_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Expected a PIL Image."
            )

        # Match Flickr30kDataset preprocessing exactly.
        image = image.convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        # Keep pixel values in [0, 255].
        # The model encoder handles its own normalization.
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

    # --------------------------------------------------
    # Service state
    # --------------------------------------------------

    @property
    def is_loaded(self):
        return (
            self.model is not None
            and self.vocabulary is not None
            and self.checkpoint_path is not None
        )


if __name__ == "__main__":
    service = ModelService()
    service.load()

    print("\nModel service is ready.")
    print("Loaded model:", service.model_type)
    print("Checkpoint:", service.checkpoint_path)
