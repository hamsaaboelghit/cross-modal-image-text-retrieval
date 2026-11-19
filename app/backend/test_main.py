
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.backend import main


class FakeRetrievalService:
    def __init__(self, loaded=True):
        self.is_loaded = loaded

    def load(self):
        return self

    def search_text(self, query, top_k=5):
        if query == "raise-error":
            raise ValueError("Invalid query.")

        return [
            {
                "type": "image",
                "score": 0.95,
                "index": 0,
                "result": "image_1.jpg",
            }
        ][:top_k]

    def search_image(self, image, top_k=5):
        return [
            {
                "type": "caption",
                "score": 0.91,
                "index": 0,
                "result": {
                    "image_name": "image_1.jpg",
                    "image_index": 0,
                    "caption_index": 0,
                    "caption": "A person walking.",
                },
            }
        ][:top_k]


@pytest.fixture
def client(monkeypatch):
    fake_service = FakeRetrievalService(loaded=True)

    monkeypatch.setattr(
        main,
        "retrieval_service",
        fake_service,
    )

    with TestClient(main.app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["backend"] == "VisionMatch AI"
    assert data["model_loaded"] is True


def test_text_search_endpoint(client):
    response = client.post(
        "/search/text",
        json={
            "query": "A dog playing in the park",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["mode"] == "text_to_image"
    assert data["top_k"] == 5
    assert len(data["results"]) == 1
    assert data["results"][0]["result"] == "image_1.jpg"


def test_text_search_uses_default_top_k(client):
    response = client.post(
        "/search/text",
        json={"query": "A dog"},
    )

    assert response.status_code == 200
    assert response.json()["top_k"] == 5


def test_text_search_rejects_empty_query(client):
    response = client.post(
        "/search/text",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_text_search_rejects_query_over_limit(client):
    response = client.post(
        "/search/text",
        json={"query": "a" * 1001},
    )

    assert response.status_code == 422


def test_text_search_rejects_invalid_top_k(client):
    response = client.post(
        "/search/text",
        json={
            "query": "A dog",
            "top_k": 51,
        },
    )

    assert response.status_code == 422


def test_text_search_maps_value_error_to_400(client):
    response = client.post(
        "/search/text",
        json={"query": "raise-error"},
    )

    assert response.status_code == 400


def test_image_search_endpoint(client):
    image = Image.new("RGB", (32, 32), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/search/image",
        files={
            "file": (
                "test.png",
                buffer.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["mode"] == "image_to_text"
    assert data["filename"] == "test.png"
    assert len(data["results"]) == 1
    assert (
        data["results"][0]["result"]["caption"]
        == "A person walking."
    )



def test_image_search_endpoint_accepts_bmp(client):
    image = Image.new("RGB", (16, 16), color="green")
    buffer = io.BytesIO()
    image.save(buffer, format="BMP")
    response = client.post(
        "/search/image",
        files={"file": ("test.bmp", buffer.getvalue(), "image/bmp")},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "image_to_text"


def test_image_search_rejects_invalid_file(client):
    response = client.post(
        "/search/image",
        files={
            "file": (
                "invalid.txt",
                b"not an image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_image_search_rejects_invalid_top_k(client):
    image = Image.new("RGB", (32, 32), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/search/image?top_k=51",
        files={
            "file": (
                "test.png",
                buffer.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 422


def test_search_returns_503_when_service_is_unavailable(monkeypatch):
    unavailable_service = FakeRetrievalService(loaded=False)

    monkeypatch.setattr(
        main,
        "retrieval_service",
        unavailable_service,
    )

    with TestClient(main.app) as test_client:
        response = test_client.post(
            "/search/text",
            json={"query": "A dog"},
        )

    assert response.status_code == 503


def test_models_endpoint_lists_selected_model(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["selected_model"] == "baseline"
    assert data["models"][0]["id"] == "baseline"
    assert data["models"][0]["loaded"] is True


def test_health_reports_degraded_when_service_missing(monkeypatch):
    monkeypatch.setattr(
        main,
        "retrieval_service",
        FakeRetrievalService(loaded=False),
    )
    with TestClient(main.app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_image_search_rejects_wrong_content_type(client):
    response = client.post(
        "/search/image",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_image_search_rejects_empty_file(client):
    response = client.post(
        "/search/image",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_text_search_rejects_whitespace_only_query(client):
    response = client.post(
        "/search/text",
        json={"query": "   "},
    )
    assert response.status_code == 422


def test_text_search_returns_rank_similarity_and_timing(client):
    response = client.post(
        "/search/text",
        json={"query": "A dog"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["rank"] == 1
    assert data["results"][0]["similarity"] == pytest.approx(0.95)
    assert data["elapsed_ms"] >= 0


def test_image_search_returns_ranked_caption_and_timing(client):
    image = Image.new("RGB", (16, 16), color="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/search/image",
        files={"file": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["rank"] == 1
    assert data["results"][0]["caption"] == "A person walking."
    assert data["elapsed_ms"] >= 0


def test_model_selection_rejects_unloaded_model(client):
    response = client.post(
        "/search/text",
        json={"query": "A dog", "model": "lstm"},
    )
    assert response.status_code == 400


def test_image_search_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 8)
    response = client.post(
        "/search/image",
        files={"file": ("large.png", b"x" * 9, "image/png")},
    )
    assert response.status_code == 413


def test_text_search_rejects_unknown_model(client):
    response = client.post(
        "/search/text",
        json={"query": "A dog", "model": "not-a-model"},
    )
    assert response.status_code == 400


def test_search_repeated_requests_remain_available(client):
    for _ in range(5):
        response = client.post(
            "/search/text",
            json={"query": "A dog"},
        )
        assert response.status_code == 200


def test_metrics_endpoint_aggregates_saved_evaluation_files(client, tmp_path, monkeypatch):
    import json

    experiment_dir = tmp_path / "lstm_fold1_seed42"
    experiment_dir.mkdir()
    metrics = {
        "experiment": "lstm_fold1_seed42",
        "model_type": "ResNet50-LSTM",
        "encoder_type": "lstm",
        "text_to_image": {"recall@1": 0.12, "recall@5": 0.34, "recall@10": 0.56},
        "image_to_text": {"recall@1": 0.15, "recall@5": 0.37, "recall@10": 0.59},
        "num_images": 5086,
        "embedding_dim": 512,
        "test_set_used": False,
    }
    (experiment_dir / "retrieval_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(main, "METRICS_PATH", tmp_path)

    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert len(payload["models"]) == 1
    assert payload["models"][0]["text_to_image"]["recall@5"] == pytest.approx(0.34)


def test_metrics_endpoint_reports_missing_results(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "METRICS_PATH", tmp_path / "missing")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_api_to_retrieval_engine_integration(tmp_path, monkeypatch):
    import json
    import numpy as np
    import tensorflow as tf
    from app.backend import retrieval_service as retrieval_module

    class TinyModelService:
        def __init__(self):
            self.model_type = "baseline"
            self.is_loaded = True
            self.checkpoint_info = {
                "checkpoint_kind": "best",
                "checkpoint_fingerprint": "checkpoint-hash",
                "vocab_fingerprint": "vocab-hash",
            }

        def load(self):
            return self

        def get_text_embedding(self, query):
            return tf.constant([[1.0] + [0.0] * 511], dtype=tf.float32)

        def get_image_embedding(self, image):
            return tf.constant([[1.0] + [0.0] * 511], dtype=tf.float32)

    image_embeddings = np.zeros((2, 512), dtype=np.float32)
    image_embeddings[0, 0] = 1.0
    image_embeddings[1, 1] = 1.0
    text_embeddings = np.zeros((2, 5, 512), dtype=np.float32)
    text_embeddings[0, :, 0] = 1.0
    text_embeddings[1, :, 1] = 1.0
    captions = [
        [f"Matching caption {index}" for index in range(5)],
        [f"Other caption {index}" for index in range(5)],
    ]
    gallery_path = tmp_path / "gallery.npz"
    np.savez_compressed(
        gallery_path,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_names=np.asarray(["match.jpg", "other.jpg"]),
        captions_json=np.asarray(json.dumps(captions)),
        metadata_json=np.asarray(json.dumps({
            "fold": 1,
            "model_type": "baseline",
            "embedding_dim": 512,
            "test_set_used": False,
            "checkpoint_fingerprint": "checkpoint-hash",
            "vocab_fingerprint": "vocab-hash",
            "metadata_version": 2,
        })),
    )
    monkeypatch.setattr(retrieval_module, "GALLERY_PATH", gallery_path)
    monkeypatch.setattr(retrieval_module, "ModelService", TinyModelService)
    real_service = retrieval_module.RetrievalService()
    monkeypatch.setattr(main, "retrieval_service", real_service)

    with TestClient(main.app) as test_client:
        text_response = test_client.post(
            "/search/text",
            json={"query": "A matching scene", "top_k": 2},
        )
        assert text_response.status_code == 200
        text_payload = text_response.json()
        assert text_payload["results"][0]["image_id"] == "match.jpg"
        assert text_payload["results"][0]["rank"] == 1

        image = Image.new("RGB", (12, 12), color="red")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_response = test_client.post(
            "/search/image",
            files={"file": ("query.png", buffer.getvalue(), "image/png")},
        )
        assert image_response.status_code == 200
        image_payload = image_response.json()
        assert image_payload["results"][0]["caption"] == "Matching caption 0"
        assert image_payload["results"][0]["rank"] == 1
