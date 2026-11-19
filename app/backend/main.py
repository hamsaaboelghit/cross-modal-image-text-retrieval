"""FastAPI application for VisionMatch AI inference.

The API loads a trained checkpoint and precomputed gallery at startup.
It never starts training in a request handler.
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from app.backend.retrieval_service import RetrievalService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_UPLOAD_BYTES = int(os.getenv("VISIONMATCH_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_QUERY_CHARS = int(os.getenv("VISIONMATCH_MAX_QUERY_CHARS", "1000"))
DEFAULT_TOP_K = int(os.getenv("VISIONMATCH_DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("VISIONMATCH_MAX_TOP_K", "50"))
IMAGE_DIR = Path(os.getenv("VISIONMATCH_IMAGE_DIR", str(PROJECT_ROOT / "data/raw/flickr30k-images"))).expanduser()
METRICS_PATH = Path(os.getenv("VISIONMATCH_METRICS_PATH", str(PROJECT_ROOT / "results/evaluation/summary.json"))).expanduser()
MODEL_TYPE = os.getenv("VISIONMATCH_MODEL_TYPE", "baseline").strip().lower()
MODEL_LABELS = {
    "baseline": "CNN + mean-pooled text baseline",
    "rnn": "ResNet50 + RNN",
    "lstm": "ResNet50 + LSTM",
    "gru": "ResNet50 + GRU",
    "transformer": "ResNet50 + Transformer",
    "cross_attention": "ResNet50 + LSTM + cross-modal attention",
}
logger = logging.getLogger("visionmatch.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

retrieval_service = RetrievalService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting VisionMatch AI API")
    try:
        retrieval_service.load()
        logger.info("Inference model and retrieval gallery loaded")
    except Exception:
        logger.exception("Inference service failed to load")
        # Keep the API available for /health and /models diagnostics.
    yield
    logger.info("Stopping VisionMatch AI API")


app = FastAPI(
    title="VisionMatch AI API",
    description=(
        "Bidirectional image-text retrieval. The API loads a pre-trained "
        "TensorFlow checkpoint and a compatible precomputed gallery at startup."
    ),
    version="1.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "System", "description": "Health and model information."},
        {"name": "Retrieval", "description": "Text-to-image and image-to-text search."},
        {"name": "Assets", "description": "Gallery image assets."},
        {"name": "Evaluation", "description": "Saved evaluation metrics."},
    ],
)

origins = [
    item.strip()
    for item in os.getenv(
        "VISIONMATCH_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RetrievalResult(BaseModel):
    type: str
    score: float
    index: int
    result: Any
    rank: int
    similarity: float
    image_id: str | None = None
    image_url: str | None = None
    caption: str | None = None


class SearchResponse(BaseModel):
    mode: str
    model: str
    top_k: int
    count: int
    elapsed_ms: float
    results: list[RetrievalResult]
    query: str | None = None
    filename: str | None = None


class TextSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Natural-language description of the target image.",
        examples=["A dog running through a green field."],
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    model: str | None = Field(
        default=None,
        description="Optional model identifier returned by GET /models.",
    )


def _service_loaded() -> bool:
    try:
        return bool(retrieval_service.is_loaded)
    except Exception:
        logger.exception("Could not inspect retrieval service state")
        return False


def _require_service(model: str | None = None) -> None:
    if not _service_loaded():
        raise HTTPException(
            status_code=503,
            detail="Inference model or retrieval gallery is not available. Check server logs and deployment assets.",
        )
    if model and model.strip().lower() != MODEL_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not loaded. Available model: '{MODEL_TYPE}'.",
        )


def _normalise_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
    result = dict(item)
    result["rank"] = rank
    result["similarity"] = float(result.get("score", 0.0))
    payload = result.get("result")
    if result.get("type") == "image" and isinstance(payload, str):
        result["image_id"] = payload
        result["image_url"] = f"/images/{Path(payload).name}"
    elif result.get("type") == "caption" and isinstance(payload, dict):
        result["image_id"] = payload.get("image_name")
        result["caption"] = payload.get("caption")
    return result


@app.get("/", tags=["System"], summary="API root")
def root():
    return {"name": "VisionMatch AI API", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["System"], summary="Check API and inference readiness")
def health_check():
    loaded = _service_loaded()
    metadata = getattr(retrieval_service, "gallery_metadata", {}) or {}
    return {
        "status": "ok" if loaded else "degraded",
        "backend": "VisionMatch AI",
        "model_loaded": loaded,
        "model_type": MODEL_TYPE,
        "gallery_loaded": bool(loaded and metadata),
    }


@app.get("/models", tags=["System"], summary="List configured inference models")
def list_models():
    """Return the model configured for this API process.

    Run a separate API process with another VISIONMATCH_MODEL_TYPE and its
    matching checkpoint/gallery to expose another model deployment.
    """
    metadata = getattr(retrieval_service, "gallery_metadata", {}) or {}
    return {
        "selected_model": MODEL_TYPE,
        "models": [{
            "id": MODEL_TYPE,
            "name": MODEL_LABELS.get(MODEL_TYPE, MODEL_TYPE.upper()),
            "loaded": _service_loaded(),
            "checkpoint_kind": getattr(
                getattr(retrieval_service, "model_service", None),
                "checkpoint_info",
                {},
            ).get("checkpoint_kind"),
            "embedding_dim": metadata.get("embedding_dim", 512),
        }],
        "selection_note": (
            "This API process serves one configured model. To compare multiple "
            "architectures, run separate configured instances or add a model registry."
        ),
    }


@app.post("/search/text", tags=["Retrieval"], summary="Retrieve images from a text query", response_model=SearchResponse)
def search_by_text(request: TextSearchRequest):
    _require_service(request.model)
    started = time.perf_counter()
    try:
        raw_results = retrieval_service.search_text(request.query, request.top_k)
        results = [_normalise_result(item, i + 1) for i, item in enumerate(raw_results)]
        return {
            "mode": "text_to_image",
            "query": request.query,
            "model": MODEL_TYPE,
            "top_k": request.top_k,
            "count": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Text retrieval failed")
        raise HTTPException(status_code=500, detail="Text retrieval failed.") from exc


@app.post("/search/image", tags=["Retrieval"], summary="Retrieve captions from an uploaded image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="PNG, JPEG, or WEBP image; maximum 10 MB by default."),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    model: str | None = Query(default=None, description="Optional model id from GET /models."),
):
    _require_service(model)
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type. Upload JPEG, PNG, or WEBP.")
    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
        try:
            with Image.open(BytesIO(image_bytes)) as opened:
                opened.verify()
            with Image.open(BytesIO(image_bytes)) as opened:
                image = opened.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=400, detail="Invalid or unsupported image file.") from exc

        started = time.perf_counter()
        raw_results = retrieval_service.search_image(image, top_k)
        results = [_normalise_result(item, i + 1) for i, item in enumerate(raw_results)]
        return {
            "mode": "image_to_text",
            "filename": file.filename,
            "model": MODEL_TYPE,
            "top_k": top_k,
            "count": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image retrieval failed")
        raise HTTPException(status_code=500, detail="Image retrieval failed.") from exc
    finally:
        await file.close()


@app.get("/images/{image_name}", tags=["Assets"], summary="Serve an image from the configured Flickr30k image directory")
def get_gallery_image(image_name: str):
    # Restrict the route to a plain filename to prevent path traversal.
    if not image_name or Path(image_name).name != image_name or image_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid image identifier.")
    root_dir = IMAGE_DIR.resolve()
    candidate = (root_dir / image_name).resolve()
    if root_dir not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image asset not found.")
    return FileResponse(candidate)


@app.get("/metrics", tags=["Evaluation"], summary="Read saved model-comparison metrics")
def get_metrics():
    path = METRICS_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return {
            "available": False,
            "message": "No final evaluation summary is available yet.",
            "source": str(path),
            "models": [],
        }
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not read evaluation metrics")
        raise HTTPException(status_code=500, detail="Saved evaluation metrics are invalid.") from exc
    return {"available": True, "source": path.name, **payload}
