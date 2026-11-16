
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from app.backend.retrieval_service import RetrievalService


# --------------------------------------------------
# Retrieval Service
# --------------------------------------------------

retrieval_service = RetrievalService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model and retrieval gallery
    when the FastAPI application starts.
    """

    print("Starting VisionMatch AI Backend...")

    retrieval_service.load()

    print("VisionMatch AI Backend is ready.")

    yield

    print("Shutting down VisionMatch AI Backend...")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="VisionMatch AI API",
    description="Cross-modal image-text retrieval backend.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request Models
# --------------------------------------------------

class TextSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural language description of the image.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of images to retrieve.",
    )


# --------------------------------------------------
# Helper
# --------------------------------------------------

def check_service():
    """
    Make sure the Retrieval Service is loaded
    before processing search requests.
    """

    if not retrieval_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Retrieval Service is not available.",
        )


# --------------------------------------------------
# Basic Endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "VisionMatch AI Backend is running."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "backend": "VisionMatch AI",
        "model_loaded": retrieval_service.is_loaded,
    }


# --------------------------------------------------
# Text-to-Image Search
# --------------------------------------------------

@app.post("/search/text")
def search_by_text(request: TextSearchRequest):
    """
    Retrieve relevant images using a text query.
    """

    check_service()

    try:
        results = retrieval_service.search_text(
            query=request.query,
            top_k=request.top_k,
        )

        return {
            "mode": "text_to_image",
            "query": request.query,
            "top_k": request.top_k,
            "results": results,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# --------------------------------------------------
# Image-to-Text Search
# --------------------------------------------------

@app.post("/search/image")
def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(
        default=5,
        ge=1,
        le=50,
    ),
):
    """
    Retrieve relevant captions using an input image.
    """

    check_service()

    try:
        image_bytes = file.file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        results = retrieval_service.search_image(
            image=image,
            top_k=top_k,
        )

        return {
            "mode": "image_to_text",
            "filename": file.filename,
            "top_k": top_k,
            "results": results,
        }

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid image.",
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
