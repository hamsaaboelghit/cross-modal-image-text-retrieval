

# VisionMatch AI
## Cross-Modal Image–Text Retrieval Using Deep Learning

### 1. Project Overview

VisionMatch AI is a deep learning project that connects images and natural language through a shared embedding space.

The system supports two main retrieval tasks:

- **Text-to-Image Retrieval:** Given a text description, retrieve the most relevant images.
- **Image-to-Text Retrieval:** Given an image, retrieve the most relevant captions.

The project explores different deep learning architectures for learning meaningful cross-modal representations and compares their retrieval performance.

### 2. Problem Definition

Traditional image retrieval systems often depend on visual features or manually assigned tags. VisionMatch AI aims to make image retrieval more flexible by allowing users to search using natural language.

The main challenge is learning a shared representation space in which semantically related images and captions are mapped close to each other.

### 3. Dataset

The project uses the **Flickr30k dataset**.

- Total images: 31,783
- Total captions: 158,915
- Captions per image: 5

The dataset is divided at the image level:

- Development set: 80% (25,426 images)
- Test set: 20% (6,357 images)

The development set is divided into five folds for cross-validation.

**Fold 1:**
- Training images: 20,340
- Validation images: 5,086

The test set remains untouched during model development and validation to prevent data leakage.

**Dataset License Notice:** The dataset images are copyrighted by their respective owners. The dataset source specifies non-commercial research and educational use. Users must follow the original dataset terms.

### 4. Data Preprocessing

The preprocessing pipeline includes:

- Image resizing to 224 × 224.
- Conversion to tensors.
- ImageNet normalization.
- Caption tokenization.
- Vocabulary construction using training captions only.
- Addition of special tokens:
  - BOS
  - EOS
  - UNK
  - PAD
- Caption encoding with a maximum sequence length of 34.

The vocabulary is built separately for each fold using only its training data.

### 5. Model Architectures

The project explores several architectures for cross-modal image–text retrieval.

#### 5.1 ResNet50 + LSTM

- ResNet50 extracts visual features from images.
- LSTM processes caption sequences.
- Attention mechanisms help the model focus on relevant information.
- Both modalities are projected into a shared 512-dimensional embedding space.

#### 5.2 GRU

- ResNet50 extracts image features.
- GRU processes text sequences.
- The model learns representations for both modalities in a shared embedding space.

#### 5.3 Transformer

- ResNet50 extracts image features.
- Transformer Encoder processes text sequences.
- Self-attention captures relationships between words.

#### 5.4 Cross-Modal Attention

This architecture extends the LSTM-based model by introducing cross-modal attention.

It allows the model to learn interactions between image and text representations before producing the final embeddings.

#### 5.5 Baseline Model

A simpler architecture is implemented as a baseline for comparison.

- Custom CNN for image feature extraction.
- Embedding layer and mean pooling for text representation.
- Projection into a shared embedding space.

### 6. Attention Mechanisms

The project implements two attention mechanisms:

**Additive Attention**

Helps the text encoder focus on relevant information in the sequence.

**Cross-Modal Attention**

Uses multi-head attention to model interactions between image and text features.

### 7. Loss Functions

The project implements multiple contrastive learning objectives:

- Multi-Positive InfoNCE Loss
- Margin-Based Contrastive Loss
- Triplet Contrastive Loss

The main experiments currently use Multi-Positive InfoNCE Loss with a temperature of 0.07.

This loss is suitable for Flickr30k because each image has five corresponding captions.

### 8. Training Pipeline

The training pipeline includes:

- PyTorch-based training.
- AdamW optimizer.
- Learning rate scheduling.
- Automatic Mixed Precision (AMP).
- Checkpoint saving.
- Early stopping support.
- Training and validation loss tracking.

Current architecture experiments use Fold 1, batch size 8, five epochs, and learning rate 0.0001.

The experiments are preliminary and use a single random seed (42).

### 9. Training Results

#### 9.1 ResNet50 + LSTM

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 2.692940 | 2.635147 |
| 2 | 2.542591 | 2.557564 |
| 3 | 2.453336 | 2.481921 |
| 4 | 2.374338 | 2.456124 |
| 5 | 2.308687 | 2.397168 |

Best validation loss: **2.397168**

#### 9.2 GRU

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 2.875395 | 2.883835 |
| 2 | 2.856948 | 2.883868 |
| 3 | 2.852591 | 2.883980 |
| 4 | 2.851523 | 2.883884 |
| 5 | 2.851102 | 2.883975 |

#### 9.3 Transformer

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 2.869546 | 2.883902 |
| 2 | 2.852899 | 2.883901 |
| 3 | 2.849560 | 2.883904 |
| 4 | 2.848843 | 2.884068 |
| 5 | 2.848328 | 2.883847 |

#### 9.4 Cross-Modal Attention

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 2.878524 | 2.869938 |
| 2 | 2.835419 | 2.838400 |
| 3 | 2.834029 | 2.844889 |
| 4 | 2.833444 | 2.805361 |
| 5 | 2.810617 | 2.796555 |

### 10. Preliminary Retrieval Evaluation

The following results were obtained on the validation set.

| Architecture | Text-to-Image R@1 | Image-to-Text R@1 |
|---|---:|---:|
| ResNet50 + LSTM | 3.496% | 5.289% |
| GRU | 0.0118% | 0.0197% |
| Transformer | 0.0118% | 0.0197% |
| Cross-Modal Attention | 0.0236% | 0.0197% |
| Baseline | Pending | Pending |

These results are preliminary.

The LSTM checkpoint was trained and resumed differently from the other architecture experiments. Therefore, the current results should not be treated as a definitive or fully fair architecture comparison.

Further experiments with consistent training settings and multiple random seeds are required.

### 11. Retrieval Evaluation Metrics

The evaluation pipeline supports:

- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Median Rank
- Mean Recall

Both retrieval directions are evaluated:

- Text-to-Image
- Image-to-Text

For Image-to-Text retrieval, all five captions belonging to the target image are considered relevant.

### 12. Retrieval Gallery

A retrieval gallery has been generated using the Fold 1 development data.

Current gallery statistics:

- Images: 25,426
- Image embeddings: 25,426
- Text embeddings: 127,130
- Embedding dimension: 512

Gallery file:

`results/retrieval_gallery_fold1/gallery.pt`

The gallery is built from development data, excluding the held-out test set.

### 13. Backend

The project includes a FastAPI backend for retrieval operations.

Available routes:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/search/text` | Text-to-Image retrieval |
| POST | `/search/image` | Image-to-Text retrieval |

Start the backend using:

```bash
python -m uvicorn app.backend.main:app --reload
```

Swagger documentation:

`http://127.0.0.1:8000/docs`

The backend has successfully started locally and loaded the model and retrieval gallery. Full endpoint integration testing is still pending.

### 14. Project Structure

```text
cross-modal-image-text-retrieval/
│
├── app/
│   └── backend/
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
│   └── evaluation/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### 15. Main Experiment Scripts

**Training:**

```text
src/training/run_baseline_experiment.py
src/training/run_gru_experiment.py
src/training/run_transformer_experiment.py
src/training/run_cross_attention_experiment.py
```

**Evaluation:**

```text
src/evaluation/evaluate_real_checkpoint.py
src/evaluation/evaluate_gru.py
src/evaluation/evaluate_transformer.py
src/evaluation/evaluate_cross_attention.py
```

### 16. Experiment Results Files

The architecture comparison results are saved in:

```text
results/experiments/architecture_comparison/
```

Files include:

- `visionmatch_model_comparison.csv`
- `visionmatch_model_comparison.json`
- `visionmatch_experiment_report.md`

### 17. Installation

Clone the repository:

```bash
git clone https://github.com/hamsaaboelghit/cross-modal-image-text-retrieval.git
```

Navigate to the project directory:

```bash
cd cross-modal-image-text-retrieval
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dataset preparation and preprocessing should be completed before running training experiments.

### 18. Current Project Status

**Implemented:**
- Flickr30k data processing pipeline.
- Image and caption preprocessing.
- Five-fold development split.
- Multiple model architectures.
- Contrastive loss implementations.
- Training and checkpoint management.
- Preliminary architecture experiments.
- Retrieval evaluation pipeline.
- Retrieval gallery generation.
- FastAPI backend startup.

**Still in progress:**
- Baseline retrieval evaluation.
- Consistent multi-seed architecture comparisons.
- Full five-fold experiments.
- Hyperparameter tuning.
- Loss function comparisons.
- Ablation studies.
- Final evaluation on the held-out test set.
- Error analysis and computational cost comparison.
- FAISS-based retrieval optimization.
- Full API endpoint testing.
- Website and application integration.

### 19. Reproducibility

The current experiments use:

- Random seed: 42
- Fold: 1
- Batch size: 8
- Epochs: 5
- Optimizer: AdamW
- Learning rate: 0.0001
- Loss: Multi-Positive InfoNCE
- Temperature: 0.07

Checkpoints, processed datasets, and experiment outputs are kept outside Git tracking where appropriate.

### 20. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Git and GitHub

---

**VisionMatch AI** aims to provide a flexible cross-modal retrieval system that connects visual information with natural language, while investigating how different deep learning architectures affect retrieval quality.

