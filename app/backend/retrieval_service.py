
from pathlib import Path

import torch
from PIL import Image

from app.backend.model_service import ModelService
from src.retrieval.retrieval_engine import RetrievalEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GALLERY_PATH = (
    PROJECT_ROOT
    / "results"
    / "retrieval_gallery_fold1"
    / "gallery.pt"
)


class RetrievalService:
    """
    Integrates the trained model, retrieval gallery,
    and RetrievalEngine.

    Supports:
        1. Text-to-Image retrieval.
        2. Image-to-Text retrieval.
    """

    def __init__(self):
        self.model_service = None
        self.retrieval_engine = None
        self.gallery_metadata = {}

    def load(self):
        """
        Load the trained model and retrieval gallery,
        then initialize the RetrievalEngine.
        """

        if self.is_loaded:
            return self

        if not GALLERY_PATH.is_file():
            raise FileNotFoundError(
                f"Retrieval gallery not found: {GALLERY_PATH}"
            )

        print("Loading Model Service...")

        self.model_service = ModelService().load()

        print("Loading Retrieval Gallery...")

        gallery = torch.load(
            GALLERY_PATH,
            map_location="cpu",
            weights_only=True,
        )

        required_keys = {
            "image_embeddings",
            "text_embeddings",
            "image_names",
            "captions",
            "metadata",
        }

        if not isinstance(gallery, dict):
            raise TypeError("Gallery must be a dictionary.")

        missing_keys = required_keys - gallery.keys()

        if missing_keys:
            raise ValueError(
                f"Gallery is missing keys: {missing_keys}"
            )

        image_embeddings = gallery["image_embeddings"]
        text_embeddings = gallery["text_embeddings"]
        image_names = gallery["image_names"]
        captions = gallery["captions"]
        metadata = gallery["metadata"]

        if metadata.get("fold") != 1:
            raise ValueError("Unexpected gallery fold.")

        if metadata.get("model_type") != "ResNet50-LSTM":
            raise ValueError("Unexpected gallery model type.")

        if metadata.get("test_set_used") is not False:
            raise ValueError(
                "Gallery must not contain test-set data."
            )

        if metadata.get("embedding_dim") != 512:
            raise ValueError(
                "Unexpected gallery embedding dimension."
            )

        print("Initializing Retrieval Engine...")

        self.retrieval_engine = RetrievalEngine(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            image_names=image_names,
            captions=captions,
        )

        self.gallery_metadata = metadata

        print("Retrieval Service loaded successfully.")

        return self

    def search_text(self, query: str, top_k: int = 5):
        """
        Retrieve images using a natural language query.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "Retrieval Service is not loaded."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        query_embedding = (
            self.model_service.get_text_embedding(query)
        )

        results = self.retrieval_engine.text_to_image(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results[0]

    def search_image(self, image: Image.Image, top_k: int = 5):
        """
        Retrieve captions using an input image.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "Retrieval Service is not loaded."
            )

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Expected a PIL Image."
            )

        query_embedding = (
            self.model_service.get_image_embedding(image)
        )

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
