# Hyperparameter Experiment Plan

## Scope

This phase performs a controlled **one-factor-at-a-time (OFAT)** search on the project's `BaselineRetrievalModel`. It is an initial hyperparameter-tuning stage, not a claim that the selected settings are optimal for every architecture. The same selected configuration can be evaluated in later architecture-specific experiments.

The experiment runner uses Flickr30k fold 1, seed 42, AdamW, weight decay 1e-4, no learning-rate scheduler, and a maximum of five epochs per run by default (with early stopping enabled). The dataset split and training pipeline are shared across the ten configurations.

## Reference configuration

| Parameter | Reference |
|---|---:|
| Learning rate | 1e-4 |
| Batch size | 8 |
| Retrieval embedding dimension | 512 |
| Dropout | 0.1 |
| InfoNCE temperature | 0.07 |

## OFAT experiment matrix

The reference configuration is run once. Each of the other nine experiments changes exactly one parameter while holding the remaining reference values fixed.

| Experiment | Changed parameter | Value |
|---|---|---:|
| reference | None | Reference |
| learning_rate_0_001 | Learning rate | 1e-3 |
| learning_rate_0_00001 | Learning rate | 1e-5 |
| batch_size_16 | Batch size | 16 |
| batch_size_32 | Batch size | 32 |
| output_dim_256 | Retrieval embedding dimension | 256 |
| dropout_0_2 | Dropout | 0.2 |
| dropout_0_3 | Dropout | 0.3 |
| temperature_0_05 | InfoNCE temperature | 0.05 |
| temperature_0_1 | InfoNCE temperature | 0.1 |

## Measurements

For each run, the script records:

- Training objective and training retrieval loss separately.
- Validation retrieval loss, including the best epoch and final epoch values.
- Text-to-image and image-to-text Recall@1, Recall@5, Recall@10, Mean Recall, MRR, and Median Rank.
- Overall validation Mean Recall and MRR.
- Training duration, epochs completed, checkpoint path, seed, fold, and configuration.

For this baseline model, the training objective and retrieval loss should be the same because the baseline does not add the auxiliary cross-modal matching objective. They are still recorded separately so the result schema remains explicit and comparable with the broader project logging.

## Selection rule and leakage prevention

The best configuration is selected using **validation overall Mean Recall** (average of text-to-image and image-to-text Mean Recall). If two runs tie, the lower best validation retrieval loss is preferred.

The test split is not used for hyperparameter tuning, ranking, early stopping, checkpoint selection, or selecting the final configuration. Test evaluation should be performed only after the model/configuration has been finalized.

## Running the experiments

List the configurations without training:

```powershell
python -m src.training.run_hyperparameter_experiments --list
```

Run all ten configurations (default maximum of five epochs each):

```powershell
python -m src.training.run_hyperparameter_experiments
```

Run selected configurations only:

```powershell
python -m src.training.run_hyperparameter_experiments --only reference batch_size_16
```

Continue the exact same run after interruption. Keep the same `--epochs` and `--only` arguments used in the original run:

```powershell
python -m src.training.run_hyperparameter_experiments --resume
```

Start a fresh run and intentionally replace the previous result files:

```powershell
python -m src.training.run_hyperparameter_experiments --overwrite
```

The resume command verifies a metadata manifest (model scope, fold, seed, epochs, and exact experiment configurations) before loading results. Completed experiments with validation metrics are skipped; failed or incomplete experiments are retried. JSON and CSV progress files are written through temporary files and atomically replaced to reduce the risk of partial result files.

## Outputs

Generated locally under `results/hyperparameter_experiments/`:

- `hyperparameter_run_metadata.json`: run identity and exact experiment plan.
- `hyperparameter_results.json`: per-experiment records.
- `hyperparameter_comparison.csv`: tabular results.
- `best_hyperparameter_configuration.json`: validation-selected configuration.
- `plots/`: validation Mean Recall and validation-loss comparisons.

Generated results, checkpoints, logs, and dataset files are runtime artifacts; they should not be committed to Git.

## Limitations

- This phase tunes the Baseline model only. It does not establish that the same hyperparameters are optimal for RNN, LSTM, GRU, Transformer, or Cross-Modal Attention architectures.
- The current matrix is an OFAT screening design, not a joint grid/random/Bayesian search. It does not measure interactions between parameters.
- The default single seed (42) is suitable for screening, not for a final statistical comparison. Promising configurations should be re-evaluated with multiple seeds in the final training phase.
- No experiment result or best configuration is claimed until the training script has actually been run and its output files inspected.
