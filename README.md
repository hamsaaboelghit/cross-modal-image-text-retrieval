
# VisionMatch AI
### Cross-Modal Image-Text Retrieval Using Deep Learning

An advanced deep learning project for **bidirectional image-text retrieval and semantic understanding**.

VisionMatch AI aims to learn a shared embedding space for images and natural language descriptions, enabling users to retrieve relevant images using text queries and find matching descriptions for input images.

The project explores CNN-based visual encoding, sequential text modeling, Transformer architectures, attention mechanisms, cross-modal interaction, and contrastive learning.

---

## Project Overview

The system follows a cross-modal retrieval pipeline:

- **Image Encoder:** Extracts visual representations from images using CNN architectures.
- **Text Encoder:** Converts natural language captions into meaningful feature representations.
- **Projection Heads:** Map image and text features into a shared embedding space.
- **Attention Mechanisms:** Focus on relevant visual and textual features.
- **Cross-Modal Attention:** Enables interaction between image regions and text tokens.
- **Contrastive Learning:** Trains the model to align matching image-caption pairs.

The project supports two retrieval directions:

1. Image-to-Text Retrieval
2. Text-to-Image Retrieval

---

## Dataset

**Flickr30k** is used as the primary dataset.

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
The project supports:

- RNN
- LSTM
- GRU
- Bidirectional RNN
- Bidirectional LSTM
- Bidirectional GRU

### 3. Transformer Architecture
A Transformer-based text encoder is implemented to model relationships between words in captions.

### 4. Advanced Cross-Modal Architecture
The advanced model extends the retrieval architecture with:

- ResNet50 visual feature extraction
- Spatial image representations
- Sequential text representations
- Multi-head Cross-Modal Attention
- Image-text matching score prediction

---

## Attention Mechanisms

The project includes:

- Additive Attention
- Multi-Head Cross-Modal Attention

These mechanisms help the model focus on relevant information across text tokens and image regions.

---

## Loss Functions

Three loss functions are implemented and integrated:

1. Multi-Positive InfoNCE Loss
2. Margin-Based Contrastive Loss
3. Triplet Contrastive Loss

Each objective is available for future comparative training experiments.

---

## Project Structure

```text
cross-modal-image-text-retrieval/
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
├── src/
│   ├── data/
│   ├── models/
│   └── training/
│
├── app/
│
├── requirements.txt
└── README.md
```

---

## Implementation & Testing Status

### Completed

- [x] Flickr30k dataset inspection and validation
- [x] Caption preparation and vocabulary construction
- [x] Five-fold data splitting and caption encoding
- [x] Image preprocessing and dataset loading
- [x] Baseline retrieval model
- [x] RNN, LSTM, and GRU text encoders
- [x] Bidirectional sequential architectures
- [x] Transformer text encoder
- [x] Additive Attention
- [x] Cross-Modal Attention
- [x] Advanced cross-modal retrieval architecture
- [x] Three contrastive loss implementations
- [x] Model configuration system
- [x] Data pipeline integration tests
- [x] Model architecture integration tests
- [x] Loss integration tests
- [x] Cross-modal model integration tests

### Verification Summary

The implemented components have passed their current integration and functionality tests, including:

- Model output shape validation
- Numerical stability checks
- Embedding normalization
- Padding and attention mask validation
- Forward and backward propagation
- Gradient existence and finiteness
- Model configuration validation

**Note:** These are implementation and integration test results, not retrieval performance measurements. Actual model training and evaluation have not yet been completed.

---

## Upcoming Work

- [ ] Training pipeline implementation
- [ ] Optimizer and learning-rate configuration
- [ ] Comparative experiments across model architectures
- [ ] Contrastive loss comparison
- [ ] Hyperparameter tuning
- [ ] Ablation studies
- [ ] Retrieval evaluation using Recall@K and other metrics
- [ ] Error analysis and computational cost analysis
- [ ] Interactive demonstration
- [ ] Final experimental report

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Scikit-learn

---

## Project Goal

The main goal is to investigate how different deep learning architectures and cross-modal learning techniques affect image-text retrieval performance, while maintaining a fair experimental setup and reliable evaluation methodology.
