
# VisionMatch AI
## Cross-Modal Image–Text Retrieval Using Deep Learning

VisionMatch AI is a deep learning project that learns a shared embedding space for images and natural language. It enables users to retrieve relevant images from text descriptions and retrieve matching captions from input images.

The project explores different neural network architectures, attention mechanisms, and contrastive learning techniques for bidirectional image–text retrieval.

> **Experimental status (November 2026):** The software pipeline is implemented and its dataset, model, evaluation, training, retrieval, backend, and website checks have been exercised locally. This does **not** mean retrieval quality is satisfactory. Current architecture results are preliminary and uneven; a short Phase 5 pilot reported validation Mean Recall of 0.0007. A controlled, reproducible comparison and final held-out test evaluation are still required. Do not interpret a working search interface or a loaded checkpoint as evidence of accurate retrieval.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Main Features](#main-features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Attention Mechanisms](#attention-mechanisms)
- [Training and Loss Functions](#training-and-loss-functions)
- [Evaluation](#evaluation)
- [Retrieval System](#retrieval-system)
- [Website Application](#website-application)
- [FastAPI Backend](#fastapi-backend)
- [Desktop Application](#desktop-application)
- [Flutter Mobile Application](#flutter-mobile-application)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Technologies Used](#technologies-used)
- [Dataset and Usage Notice](#dataset-and-usage-notice)

---

## Project Overview

Traditional image retrieval systems often depend on visual similarity or manually assigned labels. VisionMatch AI explores a more flexible approach by allowing users to search for images using natural language.

The main objective is to learn a shared semantic representation space in which matching images and captions have similar embedding vectors.

The project supports two retrieval directions:

### 1. Text-to-Image Retrieval

The user enters a natural language description, such as:

"A dog running through a green field."

The system converts the text into an embedding, compares it with the image embeddings stored in the retrieval gallery, and returns the most relevant images.

### 2. Image-to-Text Retrieval

The user uploads an image.

The system extracts its visual embedding, compares it with the stored caption embeddings, and returns the most relevant captions.

---

## Main Features

- Bidirectional image–text retrieval.
- CNN-based image feature extraction.
- Multiple text encoder architectures.
- Shared 512-dimensional embedding space.
- RNN, LSTM, GRU, and Transformer architectures.
- Additive Attention.
- Multi-head Cross-Modal Attention.
- Contrastive learning.
- Matching branch for image–caption compatibility scoring.
- Retrieval evaluation using standard ranking metrics.
- Precomputed image and caption embedding gallery.
- TensorFlow model checkpoint loading.
- FastAPI backend with text and image search endpoints.
- Automated tests for models, data pipelines, retrieval, and API functionality.

---

## Dataset

The project uses the **Flickr30k dataset**, a widely used benchmark for image–text retrieval.

| Property | Value |
|---|---:|
| Total images | 31,783 |
| Captions per image | 5 |
| Total captions | 158,915 |
| Development set | 25,426 images |
| Held-out test set | 6,357 images |

Each image has five natural-language descriptions.

### Data Splitting

The dataset is split at the image level to prevent captions belonging to the same image from leaking across different splits.

The development set is divided into five folds.

**Fold 1:**

| Split | Images |
|---|---:|
| Training | 20,340 |
| Validation | 5,086 |
| Total development | 25,426 |

The held-out test set is kept separate from training, validation, model selection, and gallery construction.

---

## Data Preprocessing

The data pipeline includes the following stages:

### Image Processing

- Load images as RGB.
- Resize images to 224 × 224 pixels.
- Convert images into TensorFlow-compatible tensors.
- Apply the appropriate image preprocessing inside the image encoder.

### Text Processing

- Clean and prepare captions.
- Build a vocabulary using training captions only.
- Convert words into integer token IDs.
- Add special tokens:
  - `PAD`
  - `UNK`
  - `BOS`
  - `EOS`
- Encode captions using a maximum sequence length of 34.

### Dataset integrity checks

Run the lightweight structural checks from the repository root:

```bash
python -m src.data.inspect_dataset
python -m src.data.validate_captions
```

The image scan reports progress every 5,000 files. To decode and verify every JPEG file separately (this can take longer on a Windows-mounted WSL drive), run:

```bash
python -m src.data.validate_images
```

These checks validate local files; the dataset itself is not stored in GitHub.

### TensorFlow Data Pipeline

The `Flickr30kDataset` class provides TensorFlow-compatible datasets using `tf.data`.

Each sample contains:

- Image tensor: `[224, 224, 3]`
- Caption tensor: `[5, 34]`
- Image filename

---

## Model Architectures

VisionMatch AI explores different architectures to compare their ability to learn meaningful image and text representations.

### 1. CNN Baseline

A lightweight baseline model provides a reference for evaluating more advanced architectures.

**Image Encoder:**
- Convolutional Neural Network.
- Convolutional feature extraction.
- Batch normalization and ReLU activation.
- Global average pooling.

**Text Encoder:**
- Trainable word embeddings.
- Mask-aware mean pooling.
- Projection into the shared embedding space.

### 2. ResNet50 + RNN

- ResNet50 extracts visual features.
- RNN processes caption sequences.
- Projection layers map image and text representations into a shared space.

### 3. ResNet50 + LSTM

- ResNet50 extracts image features.
- LSTM models sequential dependencies between words.
- Additive Attention can focus on important words.
- Both modalities are projected into 512-dimensional embeddings.

### 4. ResNet50 + GRU

- ResNet50 extracts visual features.
- GRU captures sequential information in captions.
- Image and text representations are aligned in the shared embedding space.

### 5. ResNet50 + Transformer

- ResNet50 extracts image features.
- Transformer Encoder processes caption sequences.
- Self-attention models relationships between words.

### 6. Advanced Cross-Modal Architecture

The advanced architecture extends the retrieval model with multi-head Cross-Modal Attention.

It models interactions between image regions and text sequence features and includes a matching head that predicts image–caption compatibility scores.

---

## Attention Mechanisms

### Additive Attention

Used in the text encoder to assign different importance to elements in a caption sequence.

### Cross-Modal Attention

Uses multi-head attention to model interactions between visual and textual features.

The advanced matching branch uses these fused features to estimate whether an image and caption form a matching pair.

---

## Training and Loss Functions

The training pipeline is implemented using TensorFlow and Keras.

### Training Components

- Configurable training settings.
- AdamW optimizer.
- Learning-rate configuration.
- Training and validation loss tracking.
- Model checkpoint management.
- Best-checkpoint saving.
- Early stopping.
- Reproducible random seeds.
- Training and validation pipelines.

### Loss Functions

The project includes multiple contrastive learning objectives:

- Multi-Positive InfoNCE Loss.
- Triplet Contrastive Loss.
- Margin-Based Contrastive Loss.
- Matching Binary Cross-Entropy Loss for the matching branch.

Multi-Positive InfoNCE is particularly suitable for Flickr30k because each image is associated with five positive captions.

### Initial Experiment Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow 2.16.2 |
| Fold | 1 |
| Random seed | 42 |
| Batch size | 8 |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Main retrieval loss | Multi-Positive InfoNCE |
| Temperature | 0.07 |
| Embedding dimension | 512 |

These values describe the initial baseline experiment configuration. Other experiments may use different architecture settings.

---

## Evaluation

The evaluation pipeline measures retrieval quality in both directions.

### Current model-quality status

The committed architecture-comparison report is preliminary: the LSTM results came from an earlier run, the newer GRU/Transformer/Cross-Modal Attention runs used one seed and five epochs, and the simple baseline was not included in that comparison. A later short Phase 5 `data_reference` pilot (one epoch, 50 training steps per epoch, batch size 4) reported validation Mean Recall **0.0007**. This is a diagnostic pilot, not a final benchmark, and it indicates that the current model quality needs investigation before the project can claim successful retrieval performance.

The website and API have also been smoke-tested with a locally loaded Baseline checkpoint and gallery. Both retrieval directions return results, but manual inspection found that relevance is weak. This is a functional integration result only, not an accuracy result.

Next AI work should verify checkpoint-to-gallery compatibility and embedding behavior, establish a reproducible baseline, compare selected encoders under identical data/training/evaluation settings, and repeat the selected configurations across at least three seeds. Keep the held-out test split untouched until all model and hyperparameter choices are finalized.

### Evaluation Metrics

- **Recall@1:** Percentage of queries whose relevant result appears at rank 1.
- **Recall@5:** Percentage of queries whose relevant result appears within the top 5.
- **Recall@10:** Percentage of queries whose relevant result appears within the top 10.
- **Mean Recall:** Average of the selected Recall@K values.
- **Median Rank:** Median position of the first relevant result.
- **Mean Reciprocal Rank (MRR):** Average reciprocal rank of the first relevant result.

### Retrieval Directions

**Text-to-Image:**
Each caption is used as a query to retrieve its corresponding image.

**Image-to-Text:**
Each image is used as a query to retrieve its five corresponding captions.

### Evaluation Protocol

- Validation data is used for model evaluation and selection.
- The held-out test split is not used during training or validation.
- Training and evaluation configurations are checked for consistency.
- Evaluation results are saved as JSON files under `results/experiments/`.

Example evaluation command:

```bash
python -m src.evaluation.evaluation_runner --experiment baseline --checkpoint-kind best
```

---

## Retrieval System

The retrieval system uses precomputed embeddings to perform efficient similarity-based search.

### Gallery Construction

The gallery builder:

1. Loads the trained TensorFlow model.
2. Loads the training and validation datasets.
3. Extracts image embeddings.
4. Extracts text embeddings.
5. Associates embeddings with image names and original captions.
6. Validates embedding shapes and values.
7. Saves the gallery and its metadata.

The gallery contains:

| Data | Shape |
|---|---|
| Image embeddings | `[N, 512]` |
| Text embeddings | `[N, 5, 512]` |
| Image names | `N` |
| Captions | 5 per image |

The development gallery combines training and validation images. The held-out test set is excluded.

### Gallery Output

Default output location:

```text
results/
└── retrieval_gallery_fold1/
    └── gallery.npz
```

The compressed NPZ file stores:

- `image_embeddings`
- `text_embeddings`
- `image_names`
- `captions_json`
- `metadata_json`

Build the gallery using:

```bash
python -m src.retrieval.build_gallery
```

**Important:** The combined development gallery is intended for deployment retrieval. Do not use it as a validation candidate gallery when evaluating validation queries, because it contains the validation images themselves.

---

## Website Application

A Next.js website is available for both retrieval directions. It connects to the FastAPI service and provides text/image query input, model discovery, Top-K selection, ranked results, similarity scores, loading/error states, and model insights from saved evaluation metrics. A minimum-similarity control can filter the displayed results without changing the backend ranking.

Configure the frontend API URL in `app/web/.env.local` (copy `app/web/.env.local.example`) using `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000`, then start it from the web directory:

```bash
cd app/web
npm install
npm run dev
```

Open `http://localhost:3000`. The backend must be running and healthy for live retrieval.

---

## Desktop Application

A PySide6 desktop client is available for both retrieval directions. It connects to the same FastAPI inference service as the website.

All Python dependencies, including the desktop GUI dependency, are installed from the single root requirements file:

```bash
python -m pip install -r requirements.txt
```

PySide6 is included in the same environment as the backend and training packages.

Start the backend as described below, then launch the desktop app from the repository root:

```bash
python app/desktop/main.py
```

The desktop client supports:

- Text-to-Image and Image-to-Text retrieval through the FastAPI service.
- JPEG, PNG, and WEBP uploads with a 10 MiB upload limit and local image preview.
- Top-K selection (5, 10, 20, or 50).
- Non-blocking background search and API health checks.
- Live API state (ready, degraded, or offline) with manual refresh.
- Ranked result cards with similarity scores and gallery image previews.
- Session-only search history with a clear-history action.

Install dependencies once with `python -m pip install -r requirements.txt`, then run `python app/desktop/main.py` from the repository root. Search history is held in memory only and is cleared when the application closes. The desktop app is an inference client; model loading and retrieval are handled by the backend. See [Desktop Setup Guide](app/desktop/README.md).


---

## Flutter Mobile Application

A Flutter mobile client is available under `app/mobile/flutter`. It connects to the same FastAPI inference service used by the website and desktop client. It supports Text-to-Image and Image-to-Text retrieval, photo selection from the device gallery, Top-K selection, backend readiness checks, ranked results, similarity scores, and image previews.

Flutter packages are managed separately from Python packages using `pubspec.yaml`; they do not belong in the root `requirements.txt`.

### Setup

Install Flutter 3.19 or newer and the Android SDK (Android Studio). From the Flutter app directory, generate the native runner folders and install Dart packages:

```bash
cd app/mobile/flutter
flutter create --platforms=android,ios .
flutter pub get
```

For Android development, start the API from the repository root so it listens on the local network:

```bash
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Then run the app:

```bash
cd app/mobile/flutter
flutter run
```

The app's Backend API address field defaults to `http://10.0.2.2:8000` for the Android emulator. For a physical phone, enter the computer's LAN IPv4 address (for example, `http://192.168.1.10:8000`); both devices must be on the same Wi-Fi network. Use an HTTPS URL for production. See [Flutter Mobile Setup](app/mobile/flutter/README.md) for platform-specific notes, including local HTTP and iOS photo-library configuration.

---

## FastAPI Backend

The project provides a REST API implemented using FastAPI.

The backend integrates:

- TensorFlow Model Service.
- Vocabulary loading and text tokenization.
- Image preprocessing.
- Retrieval Gallery loading.
- Retrieval Engine.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | API and inference readiness |
| GET | `/models` | Configured model and loading status |
| POST | `/search/text` | Text-to-Image retrieval |
| POST | `/search/image` | Image-to-Text retrieval |
| GET | `/images/{image_name}` | Retrieve a gallery image asset |
| GET | `/metrics` | Read saved evaluation metrics when available |

### Text Search Example

Request:

```http
POST /search/text
Content-Type: application/json
```

Body:

```json
{
  "query": "A dog running through a green field",
  "top_k": 5
}
```

The response contains the search mode, query, requested number of results, and ranked image results.

### Image Search Example

Send an image using multipart form data:

```http
POST /search/image
```

Parameters:

- `file`: Image file.
- `top_k`: Number of captions to retrieve (default: 5, maximum: 50).

### Configuration and Model Selection

Configure the model type, checkpoint directory/kind, vocabulary, gallery output/input path, image directory, CORS origins, upload limit, query length, Top-K limits, and metrics path using environment variables. Copy the root `.env.example` to `.env` and edit the values for your local artifacts. `VISIONMATCH_GALLERY_PATH` is used by both the gallery builder and API; relative paths are resolved from the repository root. See the root `.env.example` and [Backend Deployment Guide](app/backend/README.md).

A running API process serves one configured model. The optional `model` request field validates that selection; it does not hot-swap models. Multiple architectures can be exposed through separate API processes with matching checkpoints and galleries.

The `/metrics` endpoint reads a saved evaluation summary file configured through `VISIONMATCH_METRICS_PATH`. It returns an explicit unavailable response when no final metrics file exists; the frontend does not fabricate comparison scores.

### API Documentation

After starting the backend, interactive Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hamsaaboelghit/cross-modal-image-text-retrieval.git
```

### 2. Navigate to the Project

```bash
cd cross-modal-image-text-retrieval
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Main Dependencies

- Python 3.11 or 3.12
- TensorFlow 2.16.2–2.21.x (the exact allowed range is defined in requirements.txt; GPU availability depends on the operating system and CUDA/cuDNN setup)
- Keras
- NumPy
- Pandas
- Scikit-learn
- Pillow
- OpenCV
- FastAPI
- Uvicorn
- Pytest

---

## Dataset Setup

The dataset and generated model artifacts are not included as ordinary source-code files in this repository.

Prepare the Flickr30k data and place the required files in the expected project directories.

### Expected Data Directories

```text
data/
├── raw/
│   └── flickr30k-images/
│
└── processed/
    └── flickr30k/
        ├── encoded/
        │   └── fold_1/
        │       └── encoded_captions.csv
        │
        ├── vocabularies/
        │   └── fold_1_vocab.json
        │
        └── captions.csv
```

The image directory must contain the corresponding Flickr30k image files.

The processed files must match the selected fold and vocabulary configuration.

### Model Checkpoints

The backend expects a compatible TensorFlow checkpoint.

Default checkpoint directory:

```text
checkpoints/
└── experiments/
    └── baseline_fold1_seed42/
        ├── best/
        └── latest/
```

### Retrieval Gallery

The backend also requires a generated gallery:

```text
results/
└── retrieval_gallery_fold1/
    └── gallery.npz
```

Build the gallery after preparing the dataset and loading a compatible trained checkpoint.

---

## Running the Project

### 1. Configure the Model

Copy the environment template and edit it to point to the trained checkpoint and matching vocabulary:

```bash
cp .env.example .env
```

Set `VISIONMATCH_MODEL_TYPE`, `VISIONMATCH_CHECKPOINT_DIR`, and the data paths in `.env`. Checkpoints, datasets, and generated galleries remain local and are not committed to GitHub.

### 2. Generate the Retrieval Gallery

After configuring `.env` and preparing the dataset:

```bash
python -m src.retrieval.build_gallery
```

The gallery is generated from the configured model and development split at `VISIONMATCH_GALLERY_PATH` (default: `results/retrieval_gallery_fold1/gallery.npz`). New gallery metadata includes SHA-256 fingerprints of the checkpoint shards and vocabulary. The API rejects galleries with missing or mismatched fingerprints, so rebuild the gallery whenever the selected checkpoint or vocabulary changes. Build it only after selecting the final trained model; the held-out test split remains excluded.

### 3. Start the Backend

```bash
python -m uvicorn app.backend.main:app --reload --env-file .env
```

The backend starts at `http://127.0.0.1:8000`. Check `GET /health` before opening the website or desktop client; a healthy API requires a loaded checkpoint and a gallery whose checkpoint/vocabulary fingerprints match.

### 4. Open API Documentation

Swagger UI: `http://127.0.0.1:8000/docs`.

### 5. Start the Website

Configure `app/web/.env.local` using `app/web/.env.local.example`, then run:

```bash
cd app/web
npm install
npm run dev
```

Open `http://localhost:3000`. The website calls the configured FastAPI base URL and displays actual retrieval results. The model-insights page reads saved metrics from `GET /metrics` and shows an unavailable state when no evaluation summary exists.

For backend endpoint details, request/response formats, and deployment configuration, see [app/backend/README.md](app/backend/README.md).

---

## Project Structure

```text
cross-modal-image-text-retrieval/
│
├── app/
│   ├── desktop/
│   ├── mobile/
│   │   └── flutter/
│   │       ├── lib/main.dart
│   │       ├── pubspec.yaml
│   │       └── README.md
│   ├── backend/
│   │   ├── main.py
│   │   ├── model_service.py
│   │   ├── retrieval_service.py
│   │   └── README.md
│   │
│   └── web/
│       ├── src/app/
│       ├── package.json
│       └── FRONTEND_GUIDE.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
│
├── experiments/
│   └── model_configs.py
│
├── results/
│   ├── experiments/
│   └── retrieval_gallery_fold1/
│
├── src/
│   ├── data/
│   │   ├── prepare_images.py
│   │   ├── prepare_captions.py
│   │   ├── split_dataset.py
│   │   ├── build_vocabulary.py
│   │   ├── encode_captions.py
│   │   └── flickr30k_dataset.py
│   │
│   ├── models/
│   │   ├── baseline_model.py
│   │   ├── image_encoder.py
│   │   ├── text_encoder.py
│   │   ├── full_model.py
│   │   ├── cross_modal_attention.py
│   │   └── projection_head.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── experiment_runner.py
│   │   ├── train_config.py
│   │   ├── contrastive_loss.py
│   │   ├── matching_loss.py
│   │   ├── checkpoint_manager.py
│   │   └── early_stopping.py
│   │
│   ├── evaluation/
│   │   ├── evaluation_runner.py
│   │   ├── retrieval_evaluator.py
│   │   └── retrieval_metrics.py
│   │
│   └── retrieval/
│       ├── retrieval_engine.py
│       └── build_gallery.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Testing

The repository includes automated tests for data preprocessing, model components, training utilities, retrieval metrics, gallery handling, FastAPI services, and a separate optional desktop UI smoke test.

Run the focused test groups from the project root:

```bash
python -m pytest -v src/data/
python -m pytest -v src/models/
python -m pytest -v src/evaluation/
python -m pytest -v src/retrieval/
python -m pytest -v src/training/
python -m pytest -v app/backend/
```

The commands above exercise the automated unit/integration tests without intentionally launching a full training run. Run the desktop UI tests with `QT_QPA_PLATFORM=offscreen python -m pytest app/desktop/test_desktop_ui.py -v`; on headless Linux, this uses Qt's offscreen platform.

These tests validate code behavior. Tests that use mock model services, synthetic data, or synthetic galleries do not measure retrieval quality. A separate end-to-end evaluation with a trained checkpoint, matching vocabulary, and real gallery is required.

---

## System Workflow

VisionMatch AI follows a modular image–text retrieval pipeline:

1. **Dataset preparation:** Flickr30k images and captions are cleaned, split at image level, and converted into model-ready tensors and token IDs.
2. **Feature extraction:** The image encoder extracts visual features, while the selected text encoder processes caption tokens and captures their sequential or contextual information.
3. **Shared representation learning:** Projection layers map both modalities into a common 512-dimensional embedding space.
4. **Model optimization:** Contrastive objectives, including Multi-Positive InfoNCE, margin-based contrastive loss, and triplet loss, are available to train image and text representations. An optional matching branch estimates image–caption compatibility.
5. **Retrieval evaluation:** The evaluation pipeline measures ranking quality in text-to-image and image-to-text directions using Recall@K and other retrieval metrics.
6. **Gallery creation:** Image and caption embeddings are precomputed and stored with image names and captions for efficient search.
7. **Application serving:** The FastAPI backend loads a compatible trained checkpoint and retrieval gallery, converts incoming text or images into embeddings, runs similarity search, and formats ranked results.

## Deployment Notes

Training code and application serving are separate. The API loads an existing TensorFlow checkpoint; it does not start model training when a user sends a search request.

Dataset files, trained checkpoints, and generated retrieval galleries are not included as ordinary source files in the repository. They must be prepared or generated separately. The selected model checkpoint, vocabulary, preprocessing configuration, and retrieval gallery must be compatible with one another.

The default gallery is built from the development data (training and validation images) and excludes the held-out test split. For a final performance report, evaluate the selected model on the held-out test split using the project's evaluation protocol; do not use test results for model selection.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | CNN, ResNet50, OpenCV |
| Sequence Modeling | RNN, LSTM, GRU |
| Attention | Additive Attention, Multi-Head Attention |
| Data Processing | NumPy, Pandas, Pillow |
| Evaluation | Scikit-learn, TensorFlow |
| Backend | FastAPI, Uvicorn |
| Testing | Pytest |
| Version Control | Git, GitHub |

---

## Dataset and Usage Notice

Flickr30k images are copyrighted by their respective owners. This project uses the dataset for research and educational purposes.

Users are responsible for obtaining the dataset from an authorized source and following its original license, copyright, and usage conditions.

The repository contains project code and configuration files; dataset images, trained checkpoints, and generated galleries may need to be obtained or generated separately.

---

## Project Goal

VisionMatch AI aims to bridge the gap between computer vision and natural language by learning meaningful shared representations of images and text.

Through the exploration of different deep learning architectures, attention mechanisms, and contrastive learning objectives, the project provides a foundation for flexible bidirectional image–text retrieval systems.


## Evaluating a Completed Checkpoint

Evaluate only checkpoints for runs that have actually completed and whose
checkpoint files are present. At the time of this README update, the known
completed runs are the full Baseline run and the one-epoch RNN quick run.
LSTM, GRU, Transformer, and Cross-Attention must not be evaluated until their
training runs finish and produce valid checkpoints.

The evaluator uses the validation split only. It reads the fold and seed from
the checkpoint metadata, loads all five reference captions per image, and
reports Text-to-Image and Image-to-Text retrieval metrics separately. The
evaluation batch size is independent of the training batch size; it defaults
to 4 to limit memory use. It does not load the held-out test split.

### Completed Baseline checkpoint

From the repository root:

```bash
python -m src.evaluation.evaluate_checkpoint \
  --checkpoint checkpoints/experiments/allmodels_20261119_160039_baseline_fold1_seed42 \
  --model-type baseline
```

### Completed RNN checkpoint

The RNN quick run used the full model with a ResNet50 image encoder, RNN text
encoder, 300-dimensional token embeddings, 512-dimensional hidden/output
representations, one recurrent layer, dropout 0.1, pretrained ResNet50, and
no backbone unfreezing:

```bash
python -m src.evaluation.evaluate_checkpoint \
  --checkpoint checkpoints/experiments/deadline_fast_20261119_rnn_fold1_seed42 \
  --model-type full \
  --encoder-type rnn \
  --embedding-dim 300 \
  --hidden-dim 512 \
  --num-layers 1 \
  --output-dim 512 \
  --dropout 0.1 \
  --projection-dropout 0.1 \
  --pretrained \
  --no-unfreeze-last-block \
  --no-train-all-backbone \
  --no-bidirectional \
  --use-attention
```

**Architecture must match the original training run.** The current checkpoint
metadata records the training configuration but does not record every model
constructor argument. The command above reflects the known RNN quick-run
settings; do not copy it blindly for a different experiment. The evaluator
records the supplied architecture settings in its JSON output.

For a future completed LSTM, GRU, or Transformer checkpoint, change
`--encoder-type` and verify all other architecture settings against that
run's command/log before evaluating it. Do not run an evaluation command for
an unfinished or missing checkpoint.

The reported I2T Recall@K means that at least one of the five reference
captions for an image is present in the top K. It does not mean that all five
captions were retrieved. Cross-modal matching-head metrics are a separate
evaluation and must not be confused with the dual-encoder retrieval metrics.
