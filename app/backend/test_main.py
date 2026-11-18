
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


def test_image_search_rejects_invalid_file(client):
    response = client.post(
        "/search/image",
        files={
            "file": (
                "invalid.txt",
                b"not an image",
                "text/plain",
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
