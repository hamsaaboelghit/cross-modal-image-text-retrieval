# Phase 5 — Data Strategy and Pretrained vs. Scratch Results

**Project:** VisionMatch AI — Cross-Modal Image–Text Retrieval  
**Snapshot date:** 2026-11-20  
**Status:** Interim pilot report; not the final Phase 5 comparison.

## 1. Run protocol

| Setting | Value |
|---|---|
| Architecture family | ResNet50 + GRU |
| Dataset | Flickr30k, Fold 1 |
| Random seed | 42 |
| Epochs | 1 |
| Training batches per experiment | 200 |
| Validation batches | Up to 100 |
| Training batch size | 1 |
| Learning rate | 0.0001 |
| Optimizer | AdamW |
| Retrieval objective | InfoNCE |
| Temperature | 0.07 |
| Scheduler | None |
| Mixed precision | Disabled |
| Execution device | CPU (native Windows TensorFlow environment) |
| Training split | 20,340 images |
| Validation split | 5,086 images |
| Vocabulary size | 10,214 |
| Test split | Not used |

The run used a limited pilot budget rather than full-dataset training. After each completed or reused experiment, the pipeline extracted image embeddings for a 5,000-image development gallery. Gallery extraction is not a held-out test evaluation.

## 2. Results captured so far

| Experiment | Captions per image | Augmentation | ResNet50 initialization / fine-tuning | Validation retrieval loss | Training time | Status |
|---|---:|---|---|---:|---:|---|
| `gru_phase5_data_reference` | 5 | None | Pretrained; frozen | Not captured in supplied log excerpt | Reused checkpoint | Completed/reused and evaluated |
| `gru_phase5_one_caption` | 1 | None | Pretrained; frozen | 0.847462 | 7.66 min | Completed |
| `gru_phase5_light_augmentation` | 5 | Light | Pretrained; frozen | **0.805119** | 7.92 min | Completed |
| `gru_phase5_resnet50_pretrained` | 5 | None | Pretrained; last block unfrozen | Pending | Pending | Running at latest log snapshot |
| `gru_phase5_resnet50_scratch` | 5 | None | From scratch; last block unfrozen | Pending | Pending | Not yet run |

### Training-loss note

The one-caption experiment printed a training retrieval loss of 0.0000. Because the training batch size is 1, an in-batch contrastive objective has no other examples to act as negatives. Therefore, this training-loss value is degenerate and must not be used to claim that the model learned successfully. Its validation retrieval loss (0.847462) is recorded separately.

## 3. Interim observation

Of the two numerical validation retrieval-loss values currently available, light augmentation has the lower value:

- One caption: 0.847462
- Light augmentation: 0.805119

This is a provisional observation only. The experiments use a short one-epoch/200-batch budget, a single seed, and different data strategies. It is not a final controlled ranking. The pretrained fine-tuning and scratch experiments must finish before drawing conclusions about initialization.

## 4. Limitations and reproducibility

- The values above are validation retrieval losses, not test metrics.
- The held-out test split was not loaded or used.
- Only one seed (42) and one epoch were used in this pilot.
- The 200 training batches are a subset of the 20,340-image training split.
- The data-reference checkpoint was reused; its validation loss was not visible in the supplied log excerpt, so no value is inferred.
- The pretrained fine-tuning and scratch results were not available at this snapshot.
- The CPU execution environment differs from the intended GPU training setup and should be recorded when comparing runtime.

## 5. Required next update

When the remaining experiment logs are available, append their actual validation losses, retrieval metrics (Text-to-Image and Image-to-Text Recall@1/5/10, MRR, and Median Rank), training times, and checkpoint details. Update the status table and only then summarize the full Phase 5 comparison.
