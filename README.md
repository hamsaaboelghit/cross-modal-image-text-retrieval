# VisionMatch AI

### Cross-Modal Image-Text Retrieval Using Deep Learning

VisionMatch AI is a deep learning project for **bidirectional image-text retrieval and semantic representation learning**.

The system learns a shared embedding space for images and natural language descriptions, allowing users to retrieve relevant images using text queries and find matching captions for input images.

The project explores CNN-based visual encoding, sequential text modeling, Transformer architectures, attention mechanisms, cross-modal interaction, and contrastive learning.

---

## Project Overview

VisionMatch AI follows a cross-modal retrieval pipeline consisting of:

- **Image Encoder:** Extracts visual representations from images using CNN architectures.
- **Text Encoder:** Converts natural language captions into meaningful feature representations.
- **Projection Layers:** Map image and text features into a shared embedding space.
- **Attention Mechanisms:** Focus on relevant visual and textual features.
- **Cross-Modal Attention:** Enables interaction between image regions and text tokens.
- **Contrastive Learning:** Aligns matching image-caption pairs in the shared embedding space.

The system supports two retrieval directions:

1. **Text-to-Image Retrieval:** Retrieve relevant images using a natural language query.
2. **Image-to-Text Retrieval:** Retrieve matching captions for an input image.

---

## Dataset

### Flickr30k

Flickr30k is the primary dataset used for training and evaluating the image-text retrieval models.

| Statistic | Value |
|---|---:|
| Total Images | 31,783 |
| Total Captions | 158,915 |
| Captions per Image | 5 |
| Maximum Encoded Caption Length | 34 |

### Data Splitting Strategy

- 80% Development Set
- 20% Independent Test Set
- 5-Fold Cross-Validation within the Development Set

All captions belonging to the same image are kept together to prevent data leakage.

The vocabulary is built independently from each fold's training captions. Validation and test data are excluded from vocabulary construction.

### Image Preprocessing

- RGB conversion
- Resize to 224 × 224
- Tensor conversion
- ImageNet normalization

### Text Preprocessing

- Caption tokenization and encoding
- Vocabulary construction
- Padding and truncation
- Special tokens: PAD, UNK, BOS, EOS

---

## Model Architectures

### 1. Baseline Model

- CNN-based image encoder
- Embedding-based text encoder
- Mean pooling for caption representation
- Shared embedding space

### 2. Sequential Architectures

The project supports several recurrent text encoders:

- RNN
- LSTM
- GRU
- Bidirectional RNN
- Bidirectional LSTM
- Bidirectional GRU

These architectures model sequential dependencies between words in natural language captions.

### 3. Transformer Architecture

A Transformer-based text encoder is implemented to model contextual relationships between words in captions.

### 4. Advanced Cross-Modal Architecture

The advanced architecture extends the retrieval framework with:

- ResNet50 visual feature extraction
- Spatial image representations
- Sequential text representations
- Multi-Head Cross-Modal Attention
- Image-text matching score prediction

---

## Attention Mechanisms

The project includes:

- Additive Attention
- Multi-Head Cross-Modal Attention

These mechanisms help the model focus on relevant information across text tokens and image regions.

---

## Loss Functions

Three contrastive learning objectives are implemented:

1. Multi-Positive InfoNCE Loss
2. Margin-Based Contrastive Loss
3. Triplet Contrastive Loss

These objectives support comparative experiments for learning aligned image and text representations.

---

## Training Pipeline

A modular training pipeline supports model training, validation, checkpoint management, and experiment tracking.

### Training Configuration

- Configurable number of epochs and batch size
- Learning rate and weight decay
- Optimizer and scheduler selection
- Loss function selection
- Gradient clipping
- Mixed-precision training support
- Device selection (CPU/CUDA)
- Fold and random seed configuration

### Optimizers

- Adam
- AdamW
- SGD

### Learning Rate Schedulers

- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau
- No scheduler

### Training and Validation

- Modular Trainer and Evaluator components
- Training and validation loss tracking
- Gradient clipping support
- Optional CUDA automatic mixed precision
- Global training step tracking

### Checkpoint Management

- Latest checkpoint saving
- Best model checkpoint saving
- Optional per-epoch checkpoint saving
- Model, optimizer, and scheduler state restoration
- Training progress and early-stopping state restoration

### Experiment Logging

- CSV-based training logs
- Training and validation loss tracking
- Learning-rate tracking
- Epoch and training-step tracking

### Early Stopping

- Configurable patience
- Minimum improvement threshold
- Early-stopping state persistence
- Early-stopping state restoration when resuming training

---

## Retrieval System

The retrieval system uses precomputed image and text embeddings to perform similarity-based search.

### Retrieval Gallery

The Retrieval Gallery stores precomputed embeddings and their associated metadata for efficient retrieval.

For Fold 1, the gallery is constructed from the training and validation splits of the development dataset.

It contains:

- Image embeddings
- Five text embeddings per image
- Original image filenames
- Original captions
- Gallery metadata

#### Gallery Statistics

| Statistic | Value |
|---|---:|
| Development Images | 25,426 |
| Image Embeddings | 25,426 |
| Text Embeddings | 127,130 |
| Captions per Image | 5 |
| Embedding Dimension | 512 |

The embeddings are extracted using the trained ResNet50-LSTM model.

The gallery is serialized as a PyTorch file:

`results/retrieval_gallery_fold1/gallery.pt`

The independent test split is excluded from gallery construction to preserve the evaluation protocol.

### Retrieval Engine

The Retrieval Engine performs similarity-based search between input query embeddings and precomputed gallery embeddings.

It supports:

- Text-to-Image retrieval.
- Image-to-Text retrieval.
- Top-K result ranking.
- Similarity score calculation.
- Returning matching image filenames and captions.
- Preserving image-caption associations.

The engine uses normalized embeddings and dot-product similarity to rank the retrieval candidates.

### Retrieval Operations

**Text-to-Image Retrieval**

1. Receive a natural language query.
2. Convert the query into tokens using the trained vocabulary.
3. Extract a text embedding using the Text Encoder.
4. Compare the query embedding against gallery image embeddings.
5. Return the Top-K most relevant images with similarity scores.

**Image-to-Text Retrieval**

1. Receive an input image.
2. Preprocess the image using the model's image transformation pipeline.
3. Extract an image embedding using the Image Encoder.
4. Compare the query embedding against gallery text embeddings.
5. Return the Top-K matching captions with similarity scores and image metadata.

---

## Backend

The project includes a FastAPI backend that connects the trained deep learning model and retrieval system to application interfaces.

### Model Service

The Model Service is responsible for:

- Loading the trained model checkpoint.
- Loading the corresponding vocabulary.
- Preprocessing input images.
- Tokenizing and encoding text queries.
- Extracting image embeddings.
- Extracting text embeddings.
- Running inference on CPU or CUDA.

### Retrieval Service

The Retrieval Service integrates the Model Service, Retrieval Gallery, and Retrieval Engine into a unified search interface.

It provides two main operations:

- **Text-to-Image:** Converts a natural language query into a text embedding and retrieves the most relevant images.
- **Image-to-Text:** Converts an input image into an image embedding and retrieves the most relevant captions.

The service returns ranked retrieval results containing similarity scores and associated image or caption information.

The retrieval operations are exposed through:

```python
service.search_text(query, top_k=5)

service.search_image(image, top_k=5)
```

The Retrieval Service loads the trained model and gallery and initializes the Retrieval Engine for inference.

### FastAPI Endpoints

The backend exposes REST API endpoints for interacting with the retrieval system.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks whether the backend is running. |
| GET | `/health` | Returns backend status and model loading state. |
| POST | `/search/text` | Retrieves relevant images using a text query. |
| POST | `/search/image` | Retrieves matching captions using an uploaded image. |

#### 1. Root Endpoint

`GET /`

Returns a message confirming that the backend is running.

#### 2. Health Check

`GET /health`

Returns the backend status and whether the Retrieval Service is loaded.

Example response:

```json
{
    "status": "ok",
    "backend": "VisionMatch AI",
    "model_loaded": true
}
```

#### 3. Text-to-Image Search

`POST /search/text`

Accepts a JSON request containing a natural language query and the desired number of results.

**Request Body:**

```json
{
    "query": "A person riding a bicycle on the street",
    "top_k": 5
}
```

**Response:**

Returns the query, retrieval mode, requested number of results, and ranked image results with similarity scores.

Each result includes the image filename and its similarity score.

#### 4. Image-to-Text Search

`POST /search/image`

Accepts an uploaded image using `multipart/form-data` and retrieves the most relevant captions.

**Parameters:**

- `file`: Input image file (required).
- `top_k`: Number of captions to retrieve (default: 5).

**Example Request:**

```bash
curl -X POST \
  "http://127.0.0.1:8000/search/image?top_k=5" \
  -F "file=@1000092795.jpg"
```

**Response:**

Returns the uploaded filename, retrieval mode, and ranked caption results with similarity scores and associated image metadata.

### API Validation and Error Handling

The API includes input validation and error handling:

- Text queries must contain between 1 and 1000 characters.
- `top_k` must be between 1 and 50.
- Invalid image files return HTTP 400.
- Invalid parameter values return HTTP 422.
- An unavailable Retrieval Service returns HTTP 503.

### API Documentation

Interactive API documentation is available through Swagger UI:

`http://127.0.0.1:8000/docs`

The backend runs locally using Uvicorn:

```bash
python -m uvicorn app.backend.main:app --reload
```

### API Framework

- FastAPI
- Uvicorn
- CORS configuration

---

## Project Structure

```text
cross-modal-image-text-retrieval/
│
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model_service.py
│   │   └── retrieval_service.py
│   │
│   ├── web/
│   └── desktop/
│
├── checkpoints/
│   └── real_fold1/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── experiments/
│   ├── model_configs.py
│   └── test_model_configs.py
│
├── results/
│   ├── real_fold1/
│   └── retrieval_gallery_fold1/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retrieval_engine.py
│   │   └── build_gallery.py
│   │
│   └── training/
│
├── requirements.txt
└── README.md
```

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- FastAPI
- Uvicorn

---

## Project Goal

The main goal of VisionMatch AI is to investigate how different deep learning architectures and cross-modal learning techniques affect image-text retrieval performance.

The project focuses on learning meaningful shared representations, comparing visual and sequential feature extraction methods, and developing a practical bidirectional retrieval system with a reliable experimental evaluation methodology.