# Evaluation and Retrieval Audit — 2026-11-19

Repository reviewed: `hamsaaboelghit/cross-modal-image-text-retrieval`

Branch: `audit/evaluation-review-2026-11-19`

Scope: read-only review of the committed evaluation, model, dataset, training, and gallery code. No training jobs, checkpoints, datasets, or local runtime state were accessed or changed.

## Executive summary

The core retrieval-ranking implementation is structurally sound for the current Flickr30k sample layout: it evaluates Text-to-Image (T2I) and Image-to-Text (I2T) separately, and it treats all five captions belonging to an image as relevant for I2T. However, the committed evaluation runner is not configured for the latest deadline-fast experiment artifacts described by the project owner. It also does not register RNN or LSTM as standalone experiments.

The advanced cross-modal attention path is implemented as a matching branch, but the standard retrieval metrics are computed from the independent image/text embeddings produced before that branch. Therefore retrieval metrics alone do not measure the contribution of cross-modal matching.

## Findings

### 1. Retrieval evaluator: positive

File: `src/evaluation/retrieval_evaluator.py`

- T2I flattens the five captions per image into caption queries and assigns each caption its owning image as its positive candidate.
- I2T uses the five caption indices belonging to each image as its positive candidates.
- The evaluator computes both directions independently and reports Recall@1/5/10, Mean Recall, Median Rank, and MRR.
- Candidate ranking is chunked, and ties are handled deterministically by candidate index.
- Dataset extraction rejects duplicate image names and checks embedding shapes and finite values.

Important metric interpretation: I2T Recall@K in this implementation means that at least one of the five ground-truth captions appears in the top K. It is not the fraction of all five captions retrieved. The report should state this definition explicitly.

### 2. Evaluation registry is incomplete

File: `src/evaluation/evaluation_runner.py`

The committed `EXPERIMENTS` registry currently contains only:
- baseline
- gru
- transformer
- cross_attention

It does not contain standalone `rnn` or `lstm` entries, even though the text encoder supports both architectures and the project comparison includes them.

### 3. Checkpoint paths do not match the latest run naming

The committed runner expects legacy paths such as:
- `checkpoints/experiments/baseline_fold1_seed42`
- `checkpoints/experiments/gru_fold1_seed42`
- `checkpoints/experiments/transformer_fold1_seed42`
- `checkpoints/experiments/cross_attention_fold1_seed42`

The latest run paths reported by the project owner use timestamped names, including:
- `checkpoints/experiments/allmodels_20261119_160039_baseline_fold1_seed42`
- `checkpoints/experiments/deadline_fast_20261119_rnn_fold1_seed42`
- other `deadline_fast_20261119_*` architecture folders

Consequently, the current runner cannot be assumed to locate the latest checkpoints. The actual checkpoint metadata and exact final paths must be read from the laptop after each run completes.

### 4. Training/evaluation configuration is hard-coded and incompatible with fast runs

The committed evaluation runner validates a fixed training configuration (5 epochs, batch size 8, seed 42, fold 1, learning rate 1e-4, AdamW, InfoNCE, etc.). The deadline-fast runs described by the project owner used one epoch and batch size 4, with limited train/validation steps.

Those runs will correctly fail strict parity checks if evaluated as though they used the full configuration. Do not weaken or bypass the parity check just to make evaluation run. Instead, make the evaluation configuration derive from the exact checkpoint metadata and record the budget difference in every result.

The model registry also sets full-model dropout to 0.2, whereas the current fast experiment runner uses dropout 0.1. Dropout is inactive under `training=False`, so this value alone does not change inference behavior, but the architecture configuration should still match the training run for reproducibility and checkpoint auditing.

### 5. Cross-modal attention needs a separate matching evaluation

Files:
- `src/models/full_model.py`
- `src/evaluation/retrieval_evaluator.py`
- `src/evaluation/matching_evaluator.py`

The model first produces independent `image_embeddings` and `text_embeddings`. The retrieval evaluator uses only these outputs. Cross-modal attention and the matching head are applied afterward to produce `matching_scores`.

A `MatchingEvaluator` already exists and reports matching loss, accuracy, precision, recall, F1, and positive/negative score statistics. It is not called by the committed `evaluation_runner.py`.

Therefore:
- T2I/I2T metrics measure the dual-encoder retrieval path.
- Matching metrics must be reported separately for the cross-modal matching branch.
- Do not claim that the cross-modal attention improved retrieval unless a retrieval architecture actually uses its fused features or a controlled experiment demonstrates the effect.

### 6. Dataset alignment assumptions

File: `src/data/flickr30k_dataset.py`

Each dataset item is one unique image with its five captions grouped together. The evaluator preserves the order emitted by the non-shuffled validation dataset and checks duplicate image names. The I2T positive-index construction assumes the five captions remain grouped in the same image order as the image embeddings. That assumption currently matches the dataset implementation.

Before final reporting, preserve and record the exact validation image-name order for each run. Any future change to flatten captions or independently shuffle captions must also carry image-owner IDs and build positives from those IDs rather than positional assumptions.

### 7. Gallery builder avoids the held-out test split

File: `src/retrieval/build_gallery.py`

The committed gallery builder combines train and validation only and records `test_set_used: false`. This is appropriate for a development/demo gallery. The report and demo must clearly distinguish this gallery from final held-out test evaluation.

## Required remediation before final evaluation

1. Add explicit RNN and LSTM entries to the evaluation registry.
2. Add a safe checkpoint-path override so the runner can point to a completed timestamped checkpoint without renaming or moving it.
3. Build each model from the exact architecture settings recorded for that run.
4. Validate checkpoint metadata against its own recorded training configuration; do not hard-code a different epoch/batch configuration.
5. Save the run budget (epochs, steps per epoch, batch size, seed, fold, optimizer, learning rate, dropout, pretrained settings) alongside retrieval results.
6. Invoke `MatchingEvaluator` for the cross-modal model and save matching metrics separately.
7. Add a deterministic synthetic evaluator test in which each image has five distinct relevant captions, and verify both directions' expected ranks.
8. Keep validation for model selection and use the held-out test split once, after selecting the final model/configuration.

## Safety / execution notes

- This audit did not run TensorFlow, load checkpoints, build a gallery, evaluate the dataset, or modify any active training process.
- Do not overwrite, move, or delete current checkpoint directories.
- Do not compare the one-epoch fast runs directly against the five-epoch baseline as if they had equal training budgets.
- Report incomplete multi-seed experiments honestly; a single seed is not a three-seed result.
