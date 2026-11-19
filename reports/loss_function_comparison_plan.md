# Loss Function Comparison Plan

## Objective

Compare three retrieval objectives while keeping the model architecture and data pipeline fixed:

1. Multi-Positive InfoNCE
2. Margin-Based Contrastive Loss
3. Triplet Contrastive Loss

This isolates the effect of the retrieval objective more cleanly than comparing losses across different model architectures.

## Fixed experimental setup

| Setting | Value |
|---|---|
| Model | BaselineRetrievalModel |
| Dataset | Flickr30k |
| Split | Fold 1 train / validation |
| Random seed | 42 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Embedding dimension | 512 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | None |
| Default epochs | 5 (early stopping enabled) |
| InfoNCE temperature | 0.07 |

The loss comparison uses `drop_last=True` for both training and validation loss evaluation so all three objectives see identical complete batches. Retrieval metrics are computed over the validation examples emitted by that same complete-batch evaluation dataset; the evaluated image count is recorded.

## Loss-specific settings and sampling

| Loss | Setting | Positive pairs | Negative pairs |
|---|---|---|---|
| Multi-Positive InfoNCE | Temperature 0.07 | All captions belonging to each image | All captions belonging to other images in the batch |
| Margin-Based Contrastive | Margin 1.0 | All valid image-caption pairs | All valid mismatched image-caption pairs in the batch |
| Triplet | Margin 0.2 | One caption selected for each image | One caption from a different image |

Margin 1.0 is the initial setting for normalized Euclidean distances (range 0 to 2). Triplet margin 0.2 follows the current loss implementation's default.

Triplet training samples positives and in-batch negatives. Validation uses the first caption and a cyclic next-image negative, making its validation loss repeatable.

## Metrics and selection

The runner records best epoch, best/final training retrieval loss, best/final validation retrieval loss, text-to-image and image-to-text Recall@1/5/10, Mean Recall, MRR, duration, and checkpoint path.

The primary selection metric is overall validation Mean Recall. Ties are broken by lower best validation retrieval loss. The test split is not used for training, early stopping, tuning, or selection.

## Run commands

List experiments without training:

```powershell
python -m src.training.run_loss_comparison --list
```

Run all three:

```powershell
python -m src.training.run_loss_comparison
```

Run selected losses:

```powershell
python -m src.training.run_loss_comparison --only info_nce triplet
```

Resume an interrupted run using the same settings:

```powershell
python -m src.training.run_loss_comparison --resume
```

Start a fresh run and replace previous results:

```powershell
python -m src.training.run_loss_comparison --overwrite
```

## Outputs

Generated locally in `results/loss_comparison/`:
- `loss_comparison_metadata.json`
- `loss_comparison_results.json`
- `loss_comparison_results.csv`
- `best_loss_configuration.json`

Checkpoints and logs are stored under `checkpoints/experiments/loss_comparison_*` and `runs/loss_comparison/`. Runtime outputs should not be committed.

## Limitations

- The comparison uses the baseline architecture; a different architecture may favor a different loss.
- One seed and five default epochs are screening settings. Repeat promising configurations across at least three seeds before making a strong final claim.
- Margins are fixed initial settings, not separately tuned for each objective.
- Triplet uses one sampled positive and one sampled negative; InfoNCE and Margin use broader in-batch pair sets. This intrinsic difference must be considered when interpreting results.
- No winner is claimed until the training and validation outputs have actually been generated and inspected.
