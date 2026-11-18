
# VisionMatch AI
## Cross-Modal Image–Text Retrieval Using Deep Learning

VisionMatch AI is a deep learning project that learns a shared embedding space for images and natural language. It enables users to retrieve relevant images from text descriptions and retrieve matching captions from input images.

The project explores different neural network architectures, attention mechanisms, and contrastive learning techniques for bidirectional image–text retrieval.

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
- [FastAPI Backend](#fastapi-backend)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Current Implementation Status](#current-implementation-status)
- [Technologies](#technologies)
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
| GET | `/health` | Backend health check |
| POST | `/search/text` | Text-to-Image retrieval |
| POST | `/search/image` | Image-to-Text retrieval |

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

- Python 3.11
- TensorFlow 2.16.2
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

### 1. Generate the Retrieval Gallery

From the project root:

```bash
python -m src.retrieval.build_gallery
```

### 2. Start the Backend

```bash
python -m uvicorn app.backend.main:app --reload
```

The backend starts at:

```text
http://127.0.0.1:8000
```

### 3. Open API Documentation

```text
http://127.0.0.1:8000/docs
```

Use the interactive API interface to test text and image retrieval.

### Model Configuration

The backend supports environment variables for selecting the model and checkpoint paths:

| Variable | Default |
|---|---|
| `VISIONMATCH_MODEL_TYPE` | `baseline` |
| `VISIONMATCH_CHECKPOINT_DIR` | `checkpoints/experiments/baseline_fold1_seed42` |
| `VISIONMATCH_CHECKPOINT_KIND` | `best` |
| `VISIONMATCH_VOCAB_PATH` | `data/processed/flickr30k/vocabularies/fold_1_vocab.json` |

The selected checkpoint and gallery must use compatible model configurations.

---

## Project Structure

```text
cross-modal-image-text-retrieval/
│
├── app/
│   └── backend/
│       ├── main.py
│       ├── model_service.py
│       └── retrieval_service.py
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

The project includes automated tests covering:

- Dataset integrity and preprocessing.
- TensorFlow data pipelines.
- Baseline and advanced model architectures.
- RNN, LSTM, GRU, and Transformer encoders.
- Attention mechanisms.
- Gradient flow.
- Contrastive and matching loss integration.
- Retrieval metrics and ranking.
- Gallery and retrieval services.
- FastAPI endpoints.
- Package imports and training integration.

Run the complete test suite from the project root:

```bash
python -m pytest -v
```

### Latest Local Test Result

```text
103 passed
0 failed
0 errors
```

The full test suite completed successfully in the local development environment on November 18, 2026.

---

## Current Implementation Status

### Implemented and Tested

- Flickr30k dataset pipeline.
- Image and caption preprocessing.
- Vocabulary construction and caption encoding.
- CNN baseline model.
- RNN, LSTM, GRU, and Transformer text encoders.
- ResNet50-based image encoder.
- Additive Attention.
- Cross-Modal Attention.
- Matching branch and matching loss components.
- Contrastive loss implementations.
- TensorFlow training components.
- Model checkpoint loading and restoration.
- Retrieval evaluation pipeline.
- Training/evaluation configuration checks.
- Retrieval gallery construction.
- Bidirectional retrieval engine.
- Model Service.
- Retrieval Service.
- FastAPI endpoints.
- Automated test suite.

### Further Research and Development

- Complete and compare all planned training experiments.
- Improve retrieval performance through systematic experimentation.
- Run multi-seed experiments.
- Complete five-fold cross-validation.
- Compare contrastive and matching-loss configurations.
- Perform ablation studies.
- Evaluate final performance on the held-out test set.
- Conduct retrieval error analysis.
- Compare computational cost across architectures.
- Explore FAISS for large-scale retrieval optimization.
- Integrate the backend with a user-facing website or application.

Implementation status describes the current code and tested components; it does not imply that every planned experiment or final benchmark has been completed.

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
