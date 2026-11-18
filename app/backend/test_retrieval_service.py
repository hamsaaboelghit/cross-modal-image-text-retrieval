
import json

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from app.backend import retrieval_service as retrieval_module


class FakeModelService:
    def __init__(self):
        self.model_type = "baseline"
        self.is_loaded = True

    def load(self):
        return self

    def get_text_embedding(self, query):
        embedding = np.zeros((1, 512), dtype=np.float32)
        embedding[0, 0] = 1.0
        return tf.convert_to_tensor(embedding)

    def get_image_embedding(self, image):
        embedding = np.zeros((1, 512), dtype=np.float32)
        embedding[0, 1] = 1.0
        return tf.convert_to_tensor(embedding)


@pytest.fixture
def gallery_data(tmp_path, monkeypatch):
    image_embeddings = np.zeros((3, 512), dtype=np.float32)
    text_embeddings = np.zeros((3, 5, 512), dtype=np.float32)

    image_embeddings[0, 0] = 1.0
    image_embeddings[1, 1] = 1.0
    image_embeddings[2, 2] = 1.0

    for image_index in range(3):
        text_embeddings[image_index, :, image_index] = 1.0

    image_names = [
        "image_1.jpg",
        "image_2.jpg",
        "image_3.jpg",
    ]

    captions = [
        [f"Caption {j} for image 1" for j in range(5)],
        [f"Caption {j} for image 2" for j in range(5)],
        [f"Caption {j} for image 3" for j in range(5)],
    ]

    metadata = {
        "fold": 1,
        "model_type": "baseline",
        "image_count": 3,
        "captions_per_image": 5,
        "embedding_dim": 512,
        "source": "train + validation",
        "test_set_used": False,
        "format": "numpy_npz",
    }

    gallery_path = tmp_path / "gallery.npz"

    np.savez_compressed(
        gallery_path,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_names=np.asarray(image_names, dtype=str),
        captions_json=np.asarray(
            json.dumps(captions),
            dtype=str,
        ),
        metadata_json=np.asarray(
            json.dumps(metadata),
            dtype=str,
        ),
    )

    monkeypatch.setattr(
        retrieval_module,
        "GALLERY_PATH",
        gallery_path,
    )

    monkeypatch.setattr(
        retrieval_module,
        "ModelService",
        FakeModelService,
    )

    return gallery_path


def test_service_loads_gallery(gallery_data):
    service = retrieval_module.RetrievalService().load()

    assert service.is_loaded
    assert service.model_service.model_type == "baseline"
    assert service.retrieval_engine.num_images == 3
    assert service.retrieval_engine.captions_per_image == 5
    assert service.gallery_metadata["test_set_used"] is False


def test_text_to_image_search(gallery_data):
    service = retrieval_module.RetrievalService().load()

    results = service.search_text(
        "A red object",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["type"] == "image"
    assert results[0]["result"] == "image_1.jpg"


def test_image_to_text_search(gallery_data):
    service = retrieval_module.RetrievalService().load()

    image = Image.new("RGB", (224, 224), color="white")

    results = service.search_image(
        image,
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["type"] == "caption"
    assert results[0]["result"]["image_name"] == "image_2.jpg"


def test_search_requires_loaded_service():
    service = retrieval_module.RetrievalService()

    with pytest.raises(RuntimeError):
        service.search_text("A dog")


def test_empty_text_query_is_rejected(gallery_data):
    service = retrieval_module.RetrievalService().load()

    with pytest.raises(ValueError):
        service.search_text("   ")


def test_invalid_top_k_is_rejected(gallery_data):
    service = retrieval_module.RetrievalService().load()

    for invalid_k in (0, -1, 51, True):
        with pytest.raises(ValueError):
            service.search_text("A dog", top_k=invalid_k)


def test_gallery_with_test_data_is_rejected(gallery_data):
    service = retrieval_module.RetrievalService()

    (
        image_embeddings,
        text_embeddings,
        image_names,
        captions,
        metadata,
    ) = service._load_gallery()

    metadata["test_set_used"] = True

    with pytest.raises(ValueError):
        service._validate_gallery(
            image_embeddings,
            text_embeddings,
            image_names,
            captions,
            metadata,
        )


def test_gallery_with_wrong_fold_is_rejected(gallery_data):
    service = retrieval_module.RetrievalService()

    (
        image_embeddings,
        text_embeddings,
        image_names,
        captions,
        metadata,
    ) = service._load_gallery()

    metadata["fold"] = 2

    with pytest.raises(ValueError):
        service._validate_gallery(
            image_embeddings,
            text_embeddings,
            image_names,
            captions,
            metadata,
        )
