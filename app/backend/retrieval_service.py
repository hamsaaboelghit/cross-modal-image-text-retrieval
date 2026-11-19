
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from PIL import Image

from app.backend.model_service import ModelService
from src.retrieval.retrieval_engine import RetrievalEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GALLERY_PATH = Path(
    os.getenv(
        "VISIONMATCH_GALLERY_PATH",
        str(PROJECT_ROOT / "results" / "retrieval_gallery_fold1" / "gallery.npz"),
    )
).expanduser()
if not GALLERY_PATH.is_absolute():
    GALLERY_PATH = PROJECT_ROOT / GALLERY_PATH


class RetrievalService:
    """Integrates the model, retrieval gallery, and retrieval engine."""

    def __init__(self):
        self.model_service = None
        self.retrieval_engine = None
        self.gallery_metadata = {}

    def _load_gallery(self):
        if not GALLERY_PATH.is_file():
            raise FileNotFoundError(
                f"Retrieval gallery not found: {GALLERY_PATH}. "
                "Run src/retrieval/build_gallery.py first."
            )

        with np.load(GALLERY_PATH, allow_pickle=False) as gallery_file:
            required_keys = {
                "image_embeddings",
                "text_embeddings",
                "image_names",
                "captions_json",
                "metadata_json",
            }

            missing = required_keys - set(gallery_file.files)
            if missing:
                raise ValueError(f"Gallery is missing keys: {missing}")

            image_embeddings = gallery_file["image_embeddings"].copy()
            text_embeddings = gallery_file["text_embeddings"].copy()
            image_names = gallery_file["image_names"].astype(str).tolist()

            captions = json.loads(
                str(gallery_file["captions_json"].item())
            )
            metadata = json.loads(
                str(gallery_file["metadata_json"].item())
            )

        return (
            image_embeddings,
            text_embeddings,
            image_names,
            captions,
            metadata,
        )

    def _validate_gallery(
        self,
        image_embeddings,
        text_embeddings,
        image_names,
        captions,
        metadata,
    ):
        if metadata.get("fold") != 1:
            raise ValueError("Unexpected gallery fold.")

        if metadata.get("test_set_used") is not False:
            raise ValueError("Gallery must not contain test-set data.")

        if metadata.get("embedding_dim") != 512:
            raise ValueError("Unexpected gallery embedding dimension.")

        image_count = len(image_names)

        if image_embeddings.shape != (image_count, 512):
            raise ValueError(
                f"Invalid image embeddings shape: {image_embeddings.shape}"
            )

        if text_embeddings.shape != (image_count, 5, 512):
            raise ValueError(
                f"Invalid text embeddings shape: {text_embeddings.shape}"
            )

        if len(captions) != image_count:
            raise ValueError("Captions count does not match image count.")

        if len(image_names) != len(set(image_names)):
            raise ValueError("Duplicate image names detected.")

        for index, image_captions in enumerate(captions):
            if len(image_captions) != 5:
                raise ValueError(
                    f"Image {image_names[index]} must have exactly 5 captions."
                )

        if not np.isfinite(image_embeddings).all():
            raise ValueError("Non-finite image embeddings detected.")

        if not np.isfinite(text_embeddings).all():
            raise ValueError("Non-finite text embeddings detected.")

    @staticmethod
    def _validate_top_k(top_k):
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 50
        ):
            raise ValueError("top_k must be an integer between 1 and 50.")

    def load(self):
        if self.is_loaded:
            return self

        print("Loading TensorFlow Model Service...")
        self.model_service = ModelService().load()

        print("Loading Retrieval Gallery...")
        (
            image_embeddings,
            text_embeddings,
            image_names,
            captions,
            metadata,
        ) = self._load_gallery()

        self._validate_gallery(
            image_embeddings,
            text_embeddings,
            image_names,
            captions,
            metadata,
        )

        if metadata.get("model_type") != self.model_service.model_type:
            raise ValueError(
                "Gallery model type does not match the loaded model. "
                f"Gallery: {metadata.get('model_type')}, "
                f"Model: {self.model_service.model_type}."
            )

        expected_artifacts = getattr(
            self.model_service, "checkpoint_info", {}
        )
        required_fingerprints = {
            "checkpoint_fingerprint": expected_artifacts.get("checkpoint_fingerprint"),
            "vocab_fingerprint": expected_artifacts.get("vocab_fingerprint"),
        }
        missing_fingerprints = [
            key for key, expected in required_fingerprints.items()
            if not expected or not metadata.get(key)
        ]
        if missing_fingerprints:
            raise ValueError(
                "Gallery metadata is missing model compatibility fingerprints "
                f"({', '.join(missing_fingerprints)}). Rebuild the gallery with "
                "src/retrieval/build_gallery.py using the configured checkpoint "
                "and vocabulary."
            )
        for key, expected in required_fingerprints.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"Gallery {key} does not match the loaded model artifacts. "
                    "Rebuild the gallery with the currently configured checkpoint "
                    "and vocabulary."
                )

        print("Initializing Retrieval Engine...")

        self.retrieval_engine = RetrievalEngine(
            image_embeddings=tf.convert_to_tensor(
                image_embeddings,
                dtype=tf.float32,
            ),
            text_embeddings=tf.convert_to_tensor(
                text_embeddings,
                dtype=tf.float32,
            ),
            image_names=image_names,
            captions=captions,
        )

        self.gallery_metadata = metadata

        print("Retrieval Service loaded successfully.")
        return self

    def search_text(self, query: str, top_k: int = 5):
        if not self.is_loaded:
            raise RuntimeError("Retrieval Service is not loaded.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        self._validate_top_k(top_k)

        query_embedding = self.model_service.get_text_embedding(query)

        results = self.retrieval_engine.text_to_image(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results[0]

    def search_image(self, image: Image.Image, top_k: int = 5):
        if not self.is_loaded:
            raise RuntimeError("Retrieval Service is not loaded.")

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")

        self._validate_top_k(top_k)

        query_embedding = self.model_service.get_image_embedding(image)

        results = self.retrieval_engine.image_to_text(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results[0]

    @property
    def is_loaded(self):
        return (
            self.model_service is not None
            and self.model_service.is_loaded
            and self.retrieval_engine is not None
        )
