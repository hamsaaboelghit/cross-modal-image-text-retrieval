
# VisionMatch AI
## Cross-Modal Image–Text Retrieval Using Deep Learning

VisionMatch AI is a deep learning system that connects images and natural language in a shared embedding space. It allows users to search for images using text descriptions and retrieve captions that match a given image.

The project investigates different neural network architectures and contrastive learning objectives for cross-modal image–text retrieval.

## 1. Problem Definition

Traditional image retrieval systems often rely on visual similarity or manually assigned tags. This project explores a more flexible approach: retrieving images using natural language.

The main challenge is learning a shared representation space where semantically related images and captions are mapped close to each other.

The system supports two retrieval tasks:

- **Text-to-Image Retrieval:** Given a text description, retrieve the most relevant images.
- **Image-to-Text Retrieval:** Given an image, retrieve the most relevant captions.

## 2. Dataset

The project uses the **Flickr30k dataset**.

- Total images: 31,783
- Total captions: 158,915
- Captions per image: 5

### Data split

The dataset is divided at the image level to prevent images or their captions from appearing in multiple splits.

- Development set: 80% (25,426 images)
- Held-out test set: 20% (6,357 images)

The development set is divided into five folds for cross-validation.

**Fold 1:**
- Training images: 20,340
- Validation images: 5,086

The held-out test set is kept separate from training, model selection, and validation.

**Dataset License Notice:** Flickr30k images are copyrighted by their respective owners. The dataset is intended for research and educational use under its applicable terms. Users must follow the original dataset license and usage conditions.

## 3. Data Preprocessing

The preprocessing pipeline includes:

- Resizing images to 224 × 224.
- Converting images into TensorFlow tensors.
- Applying ImageNet normalization and preprocessing.
- Tokenizing captions.
- Building a vocabulary using training captions only.
- Adding special tokens: PAD, UNK, BOS, and EOS.
- Encoding captions with a maximum sequence length of 34.
- Creating TensorFlow `tf.data` pipelines for batching and training.

Vocabulary construction is performed separately for each fold using only that fold's training data.

## 4. Model Architectures

The project explores multiple architectures for learning image and text representations.

### 4.1 Baseline Model

A lightweight baseline architecture provides a reference point for evaluating more advanced models.

- CNN-based image feature extraction.
- Embedding-based text representation.
- Projection of both modalities into a shared embedding space.

### 4.2 ResNet50 + LSTM

- ResNet50 extracts visual features from images.
- LSTM processes caption sequences.
- Additive Attention can help the text encoder focus on relevant sequence information.
- Projection heads map both modalities into a shared 512-dimensional embedding space.

### 4.3 ResNet50 + GRU

- ResNet50 extracts image features.
- GRU processes caption sequences.
- Image and text representations are projected into the shared embedding space.

### 4.4 ResNet50 + Transformer

- ResNet50 extracts visual features.
- Transformer Encoder processes caption sequences.
- Self-attention models relationships between words.

### 4.5 Cross-Modal Attention

This architecture extends the image–text representation process with multi-head cross-modal attention to model interactions between visual and textual features.

## 5. Attention Mechanisms

Two attention mechanisms are implemented:

**Additive Attention**

Helps the text encoder assign different importance to elements in a caption sequence.

**Cross-Modal Attention**

Uses multi-head attention to model interactions between image and text features.

## 6. Loss Functions

The project implements multiple contrastive learning objectives:

- Multi-Positive InfoNCE Loss
- Triplet Contrastive Loss
- Margin-Based Contrastive Loss

Multi-Positive InfoNCE is particularly relevant to Flickr30k because each image has five corresponding captions.

The main experimental configuration uses a temperature of 0.07.

## 7. Training Pipeline

The training pipeline is implemented using **TensorFlow and Keras**.

It includes:

- TensorFlow-based model training.
- AdamW optimizer.
- Learning-rate scheduling.
- Training and validation loss tracking.
- Model checkpoint saving and restoration.
- Early stopping support.
- Reproducible random seeds.
- Configurable training parameters.

The initial experiment configuration uses:

| Parameter | Value |
|---|---|
| Framework | TensorFlow 2.16.2 |
| Fold | 1 |
| Batch size | 8 |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Loss | Multi-Positive InfoNCE |
| Temperature | 0.07 |
| Random seed | 42 |

These are initial experiment settings and may change during further experimentation.

## 8. Evaluation Metrics

The retrieval evaluation pipeline supports:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Median Rank
- Mean Recall

Both retrieval directions are evaluated:

- **Text-to-Image:** Each caption is used to retrieve its corresponding image.
- **Image-to-Text:** Each image is used to retrieve its five corresponding captions.

Evaluation is performed using image-caption relevance information.

## 9. Retrieval Gallery

The retrieval gallery is intended to contain image and caption embeddings generated by a trained TensorFlow model.

The gallery is built from development data only. The held-out test set must not be included.

The gallery stores:

- Image embeddings.
- Text embeddings.
- Image names.
- Captions.
- Metadata describing the model and gallery configuration.

The gallery is used by the retrieval engine to perform similarity-based searches.

## 10. Backend

The project includes a FastAPI backend for retrieval operations.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/search/text` | Text-to-Image retrieval |
| POST | `/search/image` | Image-to-Text retrieval |

Start the backend from the project root:

```bash
python -m uvicorn app.backend.main:app --reload
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

The backend requires a compatible trained TensorFlow checkpoint and a generated retrieval gallery.

## 11. Project Structure

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
├── results/
│   ├── experiments/
│   └── retrieval_gallery_fold1/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── retrieval/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 12. Main Experiment Scripts

Training scripts:

```text
src/training/run_baseline_experiment.py
src/training/run_gru_experiment.py
src/training/run_transformer_experiment.py
src/training/run_cross_attention_experiment.py
```

Retrieval gallery generation:

```text
src/retrieval/build_gallery.py
```

Evaluation and retrieval modules are organized under:

```text
src/evaluation/
src/retrieval/
```

## 13. Installation

Clone the repository:

```bash
git clone https://github.com/hamsaaboelghit/cross-modal-image-text-retrieval.git
```

Navigate to the project directory:

```bash
cd cross-modal-image-text-retrieval
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Prepare the Flickr30k dataset and run the preprocessing pipeline before starting training.

## 14. Current Project Status

### Implemented

- Flickr30k data processing pipeline.
- Image and caption preprocessing.
- Five-fold development split.
- CNN-based image encoder.
- RNN, LSTM, GRU, and Transformer text encoders.
- Additive Attention.
- Cross-Modal Attention.
- Shared image–text embedding model.
- Contrastive loss implementations.
- TensorFlow training components.
- Retrieval metrics and evaluation components.
- Retrieval engine.

### In Progress

- Full training pipeline integration.
- Consistent training of all model architectures.
- TensorFlow checkpoint validation and restoration.
- Retrieval gallery generation.
- FastAPI endpoint integration testing.
- Architecture comparison using consistent experimental settings.
- Multi-seed experiments.
- Five-fold cross-validation.
- Loss function comparisons.
- Ablation studies.
- Final evaluation on the held-out test set.
- Error analysis and computational cost comparison.
- FAISS-based retrieval optimization.
- Website and application integration.

## 15. Experimental Reproducibility

The initial experiment configuration uses:

- Random seed: 42
- Fold: 1
- Batch size: 8
- Epochs: 5
- Optimizer: AdamW
- Learning rate: 0.0001
- Loss: Multi-Positive InfoNCE
- Temperature: 0.07

For a fair architecture comparison, models should use consistent data splits, evaluation procedures, and comparable training configurations.

Final results should be reported only after the TensorFlow experiments have been completed and evaluated.

## 16. Technologies Used

- Python
- TensorFlow 2.16.2
- Keras
- NumPy
- Pandas
- Scikit-learn
- Pillow
- OpenCV
- Matplotlib
- FastAPI
- Uvicorn
- Pytest
- Git and GitHub

---

**VisionMatch AI** aims to provide a flexible cross-modal retrieval system that connects visual information with natural language while investigating how different deep learning architectures and learning objectives affect retrieval performance.
