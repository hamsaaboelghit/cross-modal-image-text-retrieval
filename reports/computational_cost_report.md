# Computational Cost Report — VisionMatch AI

**Project:** Cross-Modal Image–Text Retrieval Using Deep Learning  
**Repository:** `hamsaaboelghit/cross-modal-image-text-retrieval`  
**Status:** Living report — updated as new experiment logs and measurements are provided.  
**Last updated:** 2026-11-20

> This report records measured values only. Missing values are explicitly marked **Pending measurement**; they must not be estimated or presented as measured results. Training times from different experiment configurations are kept separate because they are not directly comparable.

## 1. What must be reported

For every model/configuration, the final computational-cost evaluation should include:

1. **Training time:** elapsed wall-clock time, with the number of epochs/steps and dataset subset stated.
2. **Inference time:** latency per image–text query and/or throughput (queries per second), with batch size and direction stated.
3. **Parameter count:** total and, ideally, trainable parameters.
4. **Model size:** saved checkpoint size on disk (and optionally serialized model size), with the measurement method stated.
5. **Hardware/software:** CPU model, GPU model and whether it was actually used, RAM, OS/runtime, TensorFlow version, precision, and batch size.

For this retrieval project, inference should be measured separately for **Text-to-Image** and **Image-to-Text**. Report both single-query latency (batch size 1) and batched throughput if feasible. Separate image-feature extraction from text-query/search time where applicable; clarify whether the image gallery embeddings are precomputed or computed online.

## 2. Current hardware inventory

| Machine | Environment currently used | Hardware facts confirmed | Important limitation |
|---|---|---|---|
| Laptop 1 | WSL/Linux, TensorFlow CPU environment | CPU execution confirmed; WSL RAM limit approximately 2.1 GB | RTX 4050 is not available to TensorFlow in this WSL environment; runs resolve to `/CPU:0` |
| Laptop 2 | Native Windows, TensorFlow 2.21.0 | TensorFlow reports no available GPUs (`GPUs []`); CPU execution | Exact CPU model and installed RAM have not yet been recorded |
| Laptop 3 | Windows, TensorFlow 2.16.2 | TensorFlow reports no available GPUs (`GPUs []`); CPU execution | Flickr30k image directory was incomplete (1,056 images); its attempted Phase 5 runs failed before valid training due to missing image `4295263796.jpg`. These are not valid timing results. |

**Hardware details still to collect:** exact CPU name/model, physical RAM, GPU name/VRAM (even if unused), OS version, and whether the GPU was used. Do not describe a GPU as the training device when the logs say `/CPU:0`.

## 3. Fair three-model comparison — Laptop 1

**Configuration shared by the three completed Seed 42 runs:**

- Dataset: Flickr30k, fold 1
- Encoder comparison: GRU, LSTM, and Cross-Attention (LSTM)
- Training: 1 epoch, 200 training steps, batch size 4
- Validation: 200 batches
- Learning rate: 0.0001; AdamW; InfoNCE; temperature 0.07
- Pretrained ResNet50 frozen; 5 captions per image; no image augmentation
- Seed: 42
- Device: TensorFlow `/CPU:0` in WSL/Linux
- Test split: **not used**

| Model/configuration | Training time | Train objective | Validation retrieval loss | Checkpoint |
|---|---:|---:|---:|---|
| GRU | 25.16 min | 2.086361 | 1.898083 | `fair200_20261120_gru_fold1_seed42` |
| LSTM | 22.03 min | 2.089612 | 1.884016 | `fair200_20261120_lstm_fold1_seed42` |
| Cross-Attention (LSTM) | 55.62 min | 2.235646 | 1.903820 | `fair200_20261120_cross_attention_fold1_seed42` |

The Cross-Attention run also enabled cross-modal matching loss (weight 0.2); GRU and LSTM did not. Therefore, the time and loss comparison is a comparison of these full configurations, not an isolated estimate of attention's computational overhead.

**Additional processing observed:** after each training run, image-feature extraction processed 5,000 images (1,250 batches at batch size 4). The extraction duration has not been separately recorded, so it is not included in the training-time column.

### Remaining fair-comparison runs

The planned comparison contains 9 runs: 3 models × 3 seeds (42, 123, 456).

| Seed | GRU | LSTM | Cross-Attention |
|---|---|---|---|
| 42 | Completed | Completed | Completed |
| 123 | Pending | Pending | Pending |
| 456 | Pending | Pending | Pending |

The three Seed 42 runs above are preliminary. Final mean and standard deviation across seeds, plus any inference and model-size comparisons, will be added when available.

## 4. Earlier architecture screening — Laptop 1

A separate initial screening used 1,000 training batches and 200 validation batches, seed 42, batch size 4, learning rate 0.0001, AdamW, InfoNCE (temperature 0.07), pretrained frozen ResNet50, and no image augmentation.

| Model | Recorded validation loss | Recorded training time |
|---|---:|---:|
| RNN | 1.56653 | 79.80 min |
| LSTM | 1.49720 | 87.84 min |
| GRU | 1.49278 | 92.23 min |
| Transformer | 2.00739 | 69.93 min |
| Cross-Attention | 1.50309 | 183.33 min |

These are screening results from a different run configuration and must **not** be mixed with the 200-step fair-comparison results above. The RNN and Transformer are included here because they were part of the initial architecture screening; whether they are part of the final model shortlist should be stated in the final report.

A later 1,500-batch continuation recorded these validation losses:
- RNN: 1.477530
- LSTM: 1.429494
- GRU: 1.425647
- Transformer: 1.852953

A Cross-Attention continuation of 500 additional batches (reported total: 1,500) recorded validation retrieval loss 1.416129 and took 111.13 minutes for that continuation. Do not add this continuation time to another run unless the exact run relationship is verified from its logs.

## 5. GRU learning-rate study — Laptop 2

**Shared setup:** GRU, seed 42, batch size 4, 200 training steps, 100 validation steps, 1 epoch per run, pretrained ResNet50, CPU-only TensorFlow 2.21.0 on native Windows.

| Learning rate | Training time | Train loss | Validation retrieval loss |
|---:|---:|---:|---:|
| 0.00005 | 12.24 min | 2.140974 | 1.987902 |
| 0.00010 | 11.99 min | 2.086361 | 1.914725 |
| 0.00020 | 11.87 min | 2.056058 | 1.815485 |

The best validation retrieval loss among these tested learning rates was observed at 0.0002. This is a small validation study, not a full-dataset final performance result.

## 6. Cross-modal attention ablation — Laptop 2

**Shared setup:** GRU, seed 42, fold 1, batch size 4, 1 epoch, 200 training steps, 100 validation steps, learning rate 0.0002, pretrained ResNet50, CPU-only TensorFlow 2.21.0 on native Windows.

| Configuration | Training time | Train objective | Validation retrieval loss |
|---|---:|---:|---:|
| Attention off | 11.35 min | 2.056058 | 1.815485 |
| Attention on | 26.51 min | 2.199328 | 1.848958 |

**Important confound:** the attention-on run also enabled cross-modal matching loss (weight 0.2), while the attention-off run did not. Therefore, this experiment changes both attention and the auxiliary matching loss; it does not isolate the effect of attention alone.

## 7. Parameters, checkpoint size, and inference latency

These measurements have not yet been supplied for the current runs.

| Model/configuration | Total parameters | Trainable parameters | Checkpoint size | Text-to-Image latency | Image-to-Text latency |
|---|---:|---:|---:|---:|---:|
| RNN | Pending | Pending | Pending | Pending | Pending |
| LSTM | Pending | Pending | Pending | Pending | Pending |
| GRU | Pending | Pending | Pending | Pending | Pending |
| Transformer | Pending | Pending | Pending | Pending | Pending |
| Cross-Attention | Pending | Pending | Pending | Pending | Pending |

For fair reporting, parameter counts should be obtained from the instantiated model architecture, and checkpoint size should be measured from the actual saved checkpoint files. Do not assume that the checkpoint size equals the in-memory parameter size.

## 8. Inference measurement protocol (to be completed)

Use the same hardware and model input dimensions for every model.

1. Load the selected checkpoint and record model-loading time separately.
2. Warm up the model before timing.
3. Measure at least 100 repeated queries (preferably more); report mean and standard deviation or median and p95 latency.
4. Measure batch size 1 latency and a fixed larger batch throughput.
5. For Text-to-Image, state whether the image gallery embeddings are precomputed. Report query-encoding/search latency separately from one-time gallery indexing.
6. For Image-to-Text, state whether caption embeddings are precomputed and report image-encoding/search latency.
7. Use the same gallery size and candidate set for all models.

## 9. Outstanding measurements and next updates

- [ ] Record exact CPU model, RAM, GPU model/VRAM, OS version, and TensorFlow version for Laptop 1 and Laptop 2.
- [ ] Finish the six remaining fair-comparison runs (Seed 123 and 456).
- [ ] Collect total and trainable parameter counts for each final model.
- [ ] Measure actual checkpoint sizes for each final model.
- [ ] Run and record Text-to-Image and Image-to-Text inference latency/throughput using a consistent protocol.
- [ ] Separately time feature extraction/gallery indexing if it is part of the deployed workflow.
- [ ] Add final mean ± standard deviation across seeds for comparable model configurations.
- [ ] Update this report with the Phase 5 data-strategy/pretrained-vs-scratch results after those experiments complete.

## 10. Reporting cautions

- Training time depends on hardware, dataset subset, batch size, number of steps/epochs, preprocessing, and whether feature extraction is included. Always report the configuration beside the time.
- Do not compare times from different laptops as if they were controlled hardware comparisons.
- Do not call validation retrieval loss an inference metric or a final test metric.
- Do not report Laptop 3's failed missing-file runs as completed experiments or valid timings.
- Keep training, feature extraction, model loading, and inference latency as separate measurements.
- Add new measurements with their source log/checkpoint and date so the report remains auditable.
