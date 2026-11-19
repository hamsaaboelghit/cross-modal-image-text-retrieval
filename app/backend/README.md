# VisionMatch AI Backend

FastAPI inference service for the cross-modal retrieval model. The API loads a trained TensorFlow checkpoint and a precomputed retrieval gallery; it does not run training during requests.

## Run

Install the repository requirements, prepare the dataset vocabulary, trained checkpoint, and matching gallery, then configure the environment variables shown in the repository-level `.env.example`.

From the repository root:

```bash
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`; the OpenAPI schema is at `/openapi.json`.

## Configuration

- `VISIONMATCH_MODEL_TYPE`: one of `baseline`, `rnn`, `lstm`, `gru`, `transformer`, or `cross_attention`.
- `VISIONMATCH_CHECKPOINT_DIR`: directory or TensorFlow checkpoint prefix for that model.
- `VISIONMATCH_CHECKPOINT_KIND`: `best` or `latest`.
- `VISIONMATCH_VOCAB_PATH`: vocabulary JSON used by the selected model.
- `VISIONMATCH_GALLERY_PATH`: matching `gallery.npz` generated for the same model, vocabulary, fold, and embedding space.
- `VISIONMATCH_IMAGE_DIR`: directory containing the Flickr30k image files used by the gallery.
- `VISIONMATCH_METRICS_PATH`: evaluation-summary JSON file or a directory containing per-experiment `retrieval_metrics.json` files (default: `results/experiments`).
- `VISIONMATCH_MAX_UPLOAD_BYTES`: maximum image upload size (default 10 MiB).
- `VISIONMATCH_CORS_ORIGINS`: comma-separated allowed frontend origins.

The current service instance loads one model at a time. To serve multiple architectures, launch separate API processes with distinct ports and model/checkpoint/gallery settings, or implement a model registry. The `model` request field is validated against the model served by the current process; it does not hot-swap TensorFlow models. The `/metrics` endpoint can aggregate the `retrieval_metrics.json` files produced under `results/experiments/<experiment>/` by the evaluation runner.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API and inference readiness |
| GET | `/models` | Configured model and loading status |
| POST | `/search/text` | Text-to-image retrieval |
| POST | `/search/image` | Image-to-text retrieval |
| GET | `/images/{image_name}` | Serve a gallery image by filename |
| GET | `/metrics` | Read a saved evaluation summary, when available |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | OpenAPI schema |

### Text-to-image

Request:

```json
{
  "query": "A dog running through a green field.",
  "top_k": 5,
  "model": "lstm"
}
```

`model` is optional and must match `VISIONMATCH_MODEL_TYPE` for this API process.

### Image-to-text

Send multipart form data with:
- `file`: JPEG, PNG, or WEBP image (default maximum 10 MiB).
- `top_k`: query parameter, integer from 1 to 50.
- `model`: optional query parameter matching the configured model.

### Response

Both search endpoints return `mode`, `model`, `top_k`, `count`, `elapsed_ms`, and ranked `results`. Each result includes `rank` and `similarity`; image results also include `image_id` and `image_url`, while caption results include `caption`.

## Error responses

- `400`: invalid query, unsupported model selection, or malformed image.
- `413`: upload exceeds the configured size limit.
- `415`: unsupported upload content type.
- `422`: request schema or top-k validation failure.
- `503`: model or gallery is not ready.
- `500`: unexpected inference or evaluation-file error.

## Gallery and checkpoint compatibility

The checkpoint, vocabulary, and gallery must come from the same trained model configuration. The gallery metadata is validated on startup, including its fold, embedding dimension, model type, and exclusion of the held-out test split. The gallery contains embeddings and captions, while the image endpoint serves the original image file from `VISIONMATCH_IMAGE_DIR`.

Do not commit large private/local checkpoints, dataset archives, or generated galleries to source control. Store them in a versioned artifact location and configure their paths per deployment.

## Tests

```bash
python -m pytest app/backend/test_main.py app/backend/test_retrieval_service.py -v
```

The API tests include mocked-service tests for request validation and response formatting. Retrieval-service tests use a synthetic gallery. These are not a substitute for a deployment smoke test using a real trained checkpoint and its matching gallery.
