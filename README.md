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
2. **Image-to-Text Retrieval:** Retrieve matching descriptions for an input image.

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

The retrieval system uses precomputed image and text embeddings to perform efficient similarity-based search.

### Retrieval Gallery

A retrieval gallery is constructed from the Fold 1 development dataset, combining the training and validation splits.

It stores:

- Image embeddings
- Five text embeddings per image
- Original image filenames
- Original captions
- Gallery metadata

The current gallery contains:

| Statistic | Value |
|---|---:|
| Development Images | 25,426 |
| Image Embeddings | 25,426 |
| Text Embeddings | 127,130 |
| Captions per Image | 5 |
| Embedding Dimension | 512 |

The gallery is generated using the trained ResNet50-LSTM model and saved locally as a PyTorch file.

**Gallery location:** `results/retrieval_gallery_fold1/gallery.pt`

The independent test split is excluded from gallery construction to preserve the evaluation protocol.

### Retrieval Engine

The Retrieval Engine performs similarity-based search using normalized embeddings.

It supports:

- Text-to-Image retrieval using text query embeddings.
- Image-to-Text retrieval using image embeddings.
- Top-K result ranking.
- Similarity score calculation.
- Returning matching filenames and captions.

The engine is designed to work with precomputed gallery embeddings rather than re-encoding the entire dataset for every search request.

---

## Backend

The project includes a FastAPI backend that provides the foundation for connecting the trained deep learning model to the application interfaces.

### Model Service

The Model Service is responsible for:

- Loading the trained model checkpoint.
- Loading the corresponding vocabulary.
- Preprocessing input images.
- Tokenizing and encoding text queries.
- Extracting image embeddings.
- Extracting text embeddings.
- Running inference on CPU or CUDA.

### API Framework

- FastAPI
- Uvicorn
- CORS configuration

The backend exposes health-check and model-service functionality as part of the application architecture.

---

## Project Structure

```text
cross-modal-image-text-retrieval/
│
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── model_service.py
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
│   │   ├── retrieval_engine.py
│   │   └── build_gallery.py
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