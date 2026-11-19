# VisionMatch AI Desktop Application

The desktop client is a PySide6 interface for the same FastAPI retrieval service used by the web application.

## 1. Install dependencies

From the repository root, activate the project virtual environment and install:

```bash
python -m pip install -r requirements-desktop.txt
```

## 2. Configure the backend

Copy the root environment template if you have not already done so:

```bash
cp .env.example .env
```

Set `VISIONMATCH_API_URL` if the API is not running at the default `http://127.0.0.1:8000`. The desktop app reads `VISIONMATCH_MODEL_TYPE` from the same root `.env` file.

The FastAPI service must be running and configured with a compatible trained checkpoint, vocabulary, Flickr30k image directory, and retrieval gallery. See [Backend Deployment Guide](../backend/README.md).

## 3. Start the desktop application

Run from the repository root:

```bash
python app/desktop/main.py
```

## Features

- Text-to-Image retrieval.
- Image-to-Text retrieval for JPEG, PNG, and WEBP uploads up to 10 MiB.
- Configurable Top-K results (5, 10, 20, or 50).
- Background API requests so the interface remains responsive.
- Clear API error messages when the backend or model artifacts are unavailable.

The desktop client does not train models. It sends inference requests to FastAPI, which loads the selected model checkpoint and matching gallery.

## Troubleshooting

- **Backend unavailable:** start FastAPI and confirm `GET /health` is reachable.
- **Inference returns 503:** check the checkpoint, vocabulary, image directory, and gallery paths in `.env`.
- **Model mismatch:** `VISIONMATCH_MODEL_TYPE` must match the model configured in the running API process.
