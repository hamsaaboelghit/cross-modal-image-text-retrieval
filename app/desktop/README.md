# VisionMatch AI Desktop Application

The desktop client is a PySide6 application for the same FastAPI retrieval service used by the website. It is an inference-only client: it does not import TensorFlow, load checkpoints, build galleries, or train models. All Python dependencies, including PySide6, are managed in the single root `requirements.txt` file.

## Install

From the repository root, activate the project virtual environment and install all project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configure the backend

Copy the root environment template if needed:

```bash
cp .env.example .env
```

Set `VISIONMATCH_API_URL` to the FastAPI base URL (default `http://127.0.0.1:8000`) and `VISIONMATCH_MODEL_TYPE` to the model served by that API process. The API must be configured with a compatible checkpoint, vocabulary, image directory, and fingerprint-matched retrieval gallery. See [Backend Deployment Guide](../backend/README.md).

## Run

From the repository root:

```bash
python app/desktop/main.py
```

## Features

- Text-to-Image retrieval from natural-language descriptions.
- Image-to-Text retrieval from JPEG, PNG, and WEBP uploads.
- Upload size validation (10 MiB maximum) and image preview.
- Configurable Top-K (5, 10, 20, or 50).
- Background search and API health requests so the UI remains responsive.
- API readiness state distinguishing offline, degraded, and model-ready states.
- Ranked result cards with similarity scores, text details, and image thumbnails fetched asynchronously from the gallery.
- Session-only search history with a clear-history action.
- About page describing the dataset, model family, and inference architecture.
- Clear handling of HTTP errors, unavailable services, and malformed responses.

Search history is kept in memory for the current application session only; it is not persisted to disk.

## Tests

Run the UI smoke tests without opening a visible desktop window:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest app/desktop/test_desktop_ui.py -v
```

The desktop CI workflow installs the optional Qt dependencies, compiles the package, and runs these offscreen tests.

## Troubleshooting

- **API offline:** start FastAPI and verify `GET /health` at the configured API URL.
- **API online but model unavailable:** inspect the API's `startup_error`; check the checkpoint, vocabulary, and gallery paths and rebuild a gallery if its fingerprints do not match.
- **Model mismatch:** `VISIONMATCH_MODEL_TYPE` must match the model configured in the running API process.
- **Wrong API URL:** update `VISIONMATCH_API_URL` in the root `.env` file and restart the desktop app.
