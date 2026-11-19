# Model Comparison Audit – VisionMatch AI

## 1. Purpose

This document audits the five planned model configurations for the Cross-Modal Image-Text Retrieval project.

The goal is to verify their architectures, shared training settings, differences, and readiness for a fair comparison.

**Important:** Having an experiment script does not prove that the experiment has completed or produced valid results.

## 2. Model Architecture Comparison

| Model | Image Encoder | Text Encoder | Pretrained Image Weights | Cross-Modal Attention |
|---|---|---|---|---|
| Baseline | Custom CNN | Embedding + Mean Pooling | No | No |
| LSTM | ResNet50 | LSTM | ImageNet | No |
| GRU | ResNet50 | GRU | ImageNet | No |
| Transformer | ResNet50 | Transformer | ImageNet | No |
| Cross-Modal Attention | ResNet50 | LSTM | ImageNet | Yes |

### Architecture details

- **Baseline:** Uses a custom CNN for image features and Mean Pooling for text features.
- **LSTM:** Uses ResNet50 for image features and an LSTM-based text encoder.
- **GRU:** Uses ResNet50 and a GRU-based text encoder.
- **Transformer:** Uses ResNet50 and a Transformer-based text encoder.
- **Cross-Modal Attention:** Uses the same ResNet50 + LSTM foundation, with an additional cross-modal attention and matching branch.

The full models use a 300-dimensional token embedding and a 512-dimensional hidden representation by default. The retrieval embedding dimension is configurable through `output_dim` (default: 512) in `CrossModalRetrievalModel`.

## 3. Verified Training Configuration

The local `experiment_runner.py` and experiment scripts confirm the following settings:

| Setting | Value |
|---|---|
| Dataset | Flickr30k |
| Fold | 1 |
| Random Seed | 42 |
| Epochs | 5 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Scheduler | None |
| Retrieval Loss | Multi-Positive InfoNCE |
| Temperature | 0.07 |
| Text Token Embedding | 300 |
| Hidden Dimension | 512 |
| Full-model Retrieval Embedding | 512 |
| ResNet50 Pretrained | Yes |
| ResNet50 Fine-tuning | Disabled |
| Mixed Precision | Disabled |

All five architecture-comparison runs now use an explicit dropout value of 0.1. The shared runner also defaults to 0.1 for both Baseline and full models, so an omitted dropout argument cannot silently introduce a different rate.

The experiment scripts use the same fold, seed, epoch count, and batch size.

## 4. Cross-Modal Attention Configuration

The dedicated script is:

`src/training/run_cross_attention_experiment.py`

Its configuration is:

- Image Encoder: ResNet50
- Text Encoder: LSTM
- Pretrained: True
- Unfreeze Last Block: False
- Cross-Modal Attention: True
- Epochs: 5
- Batch Size: 8
- Seed: 42
- Fold: 1

Unlike the other four experiments, this model includes an additional matching branch.

The training objective for Cross-Modal Attention includes an auxiliary matching loss with a default weight of 0.2. The other architectures optimize retrieval loss only. This objective difference remains intentional and must be disclosed.

For validation and model selection, however, every architecture now uses the same retrieval-only validation loss. The matching validation loss and combined validation objective are logged separately for diagnosis, but they do not control early stopping, scheduler decisions, or best-checkpoint selection.

## 5. Fair Comparison Audit

### What is currently consistent?

- All five experiment scripts use the same dataset fold.
- All five use the same random seed.
- All five specify the same epoch count, batch size, and dropout rate (0.1).
- All five use the common experiment runner.
- All five use retrieval-only validation loss for early stopping and best-checkpoint selection.
- The four full models use the same pretrained, frozen ResNet50 backbone.
- The four full models use a 512-dimensional retrieval embedding.

### Differences that must be documented

**1. Different image encoders**

The Baseline uses a custom CNN, while the other models use ResNet50 pretrained on ImageNet.

**2. Training objective difference**

The Cross-Modal Attention model adds a matching loss to the retrieval loss.

**3. Different model capacity**

The models have different architectures and parameter counts. This is expected, but should be documented when interpreting results.

**4. Retrieval embedding configuration**

`CrossModalRetrievalModel` accepts `output_dim` (default 512), and `experiment_runner.py` passes the configured value to the full-model constructor. The model uses this value for both image and text projection outputs. An automated regression test now checks that setting `output_dim=256` produces image embeddings shaped `[B, 256]` and text embeddings shaped `[B, C, 256]`.

## 6. Data and Evaluation Notes

The runner loads the training and validation splits only.

The test split is not passed to the training process, which is appropriate for keeping the test set separate from model selection.

The dataset supplies RGB images in the range [0, 255]. The ResNet50 image encoder applies its own ResNet50 preprocessing internally.

The Baseline and full models use different image architectures and potentially different preprocessing paths. This should be checked and documented as part of the final comparison.

## 7. Current Status

| Requirement | Status |
|---|---|
| Baseline experiment script | Verified |
| LSTM experiment script | Verified |
| GRU experiment script | Verified |
| Transformer experiment script | Verified |
| Cross-Modal Attention experiment script | Verified |
| Common training settings (including dropout) | Verified in code |
| Model architecture differences | Documented |
| Cross-modal matching loss difference | Identified and separately logged |
| Actual completion of all five experiments | Not yet verified |
| Training logs and checkpoints | Pending audit |
| Validation retrieval metrics | Pending audit |
| Test-set evaluation | Pending audit |
| Final performance comparison | Pending |

## 8. Conclusion

The repository contains five experiment configurations covering a Baseline, three sequence-model alternatives, and an additional Cross-Modal Attention model.

The training scripts now share the selected dropout rate and retrieval-only validation model-selection criterion. The comparison still contains documented differences in image encoders and training objectives, and it is not complete until preprocessing parity, experiment logs, checkpoints, and validation retrieval metrics have been verified.

The next step is to verify the actual experiment outputs, logs, and checkpoints before claiming that all five models have been trained and compared.

No performance conclusions should be made until the validation results are collected and evaluated consistently.