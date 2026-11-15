
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="VisionMatch AI API",
    description="Cross-modal image-text retrieval backend.",
    version="1.0.0",
)

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
        "model_loaded": False,
    }
