# Computational Cost Analysis

**Project:** VisionMatch AI — Cross-Modal Image–Text Retrieval Using Deep Learning  
**Report type:** Academic project report  
**Status:** Preliminary; incomplete measurements are explicitly marked as pending.  
**Last updated:** 20 November 2026

## 1. Purpose and scope

This section evaluates the computational requirements of the image–text retrieval models. The comparison must consider not only retrieval quality but also the resources required to train, store, and use each model.

For every model included in the final comparison, the report will provide:

- **Training time:** elapsed time required for the stated training configuration.
- **Inference time:** time required to process a query and retrieve matching items.
- **Parameter count:** total number of model parameters and, where relevant, the number of trainable parameters.
- **Model size:** storage occupied by the saved model/checkpoint.
- **Hardware:** processor, graphics processor (if used), memory, and relevant software/runtime details.

All measurements must be accompanied by their experimental conditions. Values obtained with different datasets, step counts, batch sizes, hardware, or training configurations must not be treated as directly comparable.

## 2. Training time

Training time is the wall-clock duration required to train a model under a clearly specified configuration. It should be reported together with the number of epochs, training steps, batch size, dataset split/subset, optimizer, and hardware.

### 2.1 Preliminary three-model comparison

The following results are currently available for the three main model configurations. Each run used Flickr30k fold 1, one epoch, 200 training steps, 200 validation batches, batch size 4, learning rate 0.0001, AdamW, InfoNCE loss with temperature 0.07, a pretrained frozen ResNet50 image encoder, five captions per image, no image augmentation, and random seed 42. Training ran on the CPU.

| Model configuration | Training time | Training objective | Validation retrieval loss |
|---|---:|---:|---:|
| GRU text encoder | 25.16 min | 2.086361 | 1.898083 |
| LSTM text encoder | 22.03 min | 2.089612 | 1.884016 |
| LSTM with cross-modal attention | 55.62 min | 2.235646 | 1.903820 |

**Interpretation and limitation:** In this preliminary run, the LSTM configuration had the shortest recorded training time. The cross-modal attention configuration took longer. However, its training setup also enabled an auxiliary cross-modal matching loss (weight 0.2), whereas the GRU and LSTM runs did not. Consequently, these values compare the complete configurations and do not isolate the computational overhead caused by attention alone.

The comparison is preliminary because only seed 42 is currently available. The remaining runs for seeds 123 and 456 are **Pending**. Final training-time statistics should be reported as mean and standard deviation across the completed repeated runs, where the configurations are comparable.

### 2.1.1 Repeated-seed comparison — additional GRU result

A second GRU run has now completed for seed 123 under the same fold-1, one-epoch, 200-training-step, 200-validation-batch, batch-size-4 configuration described above.

| Model | Seed | Training time | Training objective | Validation retrieval loss |
|---|---:|---:|---:|---:|
| GRU | 123 | 25.51 min | 2.039485 | 1.826844 |

The seed-123 validation loss is lower than the seed-42 GRU result (1.898083). This reinforces the need to complete all three seeds before reporting mean ± standard deviation or selecting a final architecture. Seed-123 LSTM and cross-modal attention, and all seed-456 runs, remain **Pending**. Test data was not used.

### 2.2 Additional training-time experiments

The following supplementary measurements are available. They are separate experiments and must not be merged with the three-model comparison above.

**GRU learning-rate study** — one epoch, 200 training steps, 100 validation batches, batch size 4:

| Learning rate | Training time | Training loss | Validation retrieval loss |
|---:|---:|---:|---:|
| 0.00005 | 12.24 min | 2.140974 | 1.987902 |
| 0.00010 | 11.99 min | 2.086361 | 1.914725 |
| 0.00020 | 11.87 min | 2.056058 | 1.815485 |

**Cross-modal attention ablation** — GRU, one epoch, 200 training steps, 100 validation batches, batch size 4, learning rate 0.0002:

| Configuration | Training time | Training objective | Validation retrieval loss |
|---|---:|---:|---:|
| Attention disabled | 11.35 min | 2.056058 | 1.815485 |
| Attention enabled | 26.51 min | 2.199328 | 1.848958 |

**Ablation limitation:** The attention-enabled experiment also enabled the auxiliary cross-modal matching loss (weight 0.2), while the attention-disabled experiment did not. This is therefore not a perfectly isolated attention ablation.

### 2.3 Other architecture screening results

An earlier screening experiment used 1,000 training batches and 200 validation batches, with seed 42 and batch size 4. Its results are shown separately because the training budget differs from the preliminary three-model comparison.

| Architecture | Training time | Validation retrieval loss |
|---|---:|---:|
| RNN | 79.80 min | 1.56653 |
| LSTM | 87.84 min | 1.49720 |
| GRU | 92.23 min | 1.49278 |
| Transformer | 69.93 min | 2.00739 |
| Cross-Attention | 183.33 min | 1.50309 |

These screening results are exploratory and should not be compared directly with the one-epoch, 200-step runs.

## 3. Inference time

Inference time measures how long a trained model takes to process input and return retrieval results. It is different from training time.

Because VisionMatch AI supports two retrieval directions, inference must be measured separately:

1. **Text-to-Image retrieval:** encode a text query and retrieve the most relevant images.
2. **Image-to-Text retrieval:** encode an input image and retrieve the most relevant captions.

The final report should include:

- Single-query latency with batch size 1.
- Throughput for a stated batch size, such as queries per second.
- Mean latency and a variability measure (standard deviation, median, or p95).
- Model-loading time, reported separately from query inference.
- Whether candidate image/caption embeddings are precomputed or generated at query time.
- Gallery/index size used during measurement.

For a fair comparison, all models must be measured using the same hardware, input dimensions, candidate gallery, batch sizes, and timing procedure. Image-gallery feature extraction or indexing should be measured separately from query-time inference if embeddings are precomputed.

**Current status:** Text-to-Image inference time — **Pending measurement**. Image-to-Text inference time — **Pending measurement**. Throughput and latency statistics — **Pending measurement**.

## 4. Model parameter count

The parameter count indicates the number of numerical values learned or used by the model. The report should provide:

- **Total parameters:** all parameters in the instantiated model.
- **Trainable parameters:** parameters updated during training.
- **Non-trainable parameters:** parameters held fixed, such as a frozen pretrained image encoder.

This distinction is important because the project uses a pretrained ResNet50 image encoder, which may be frozen in some experiments and fine-tuned in others. The total parameter count can remain the same while the trainable parameter count changes.

Parameter counts should be obtained from the actual instantiated model for each architecture and configuration, rather than estimated from architecture names.

| Model | Total parameters | Trainable parameters |
|---|---:|---:|
| RNN | Pending measurement | Pending measurement |
| LSTM | Pending measurement | Pending measurement |
| GRU | Pending measurement | Pending measurement |
| Transformer | Pending measurement | Pending measurement |
| Cross-modal attention model | Pending measurement | Pending measurement |

## 5. Model size

Model size refers to the storage required by the saved trained model.

The final report should state the actual size of the saved checkpoint or exported model file, measured in MB. If training checkpoints contain optimizer state or other metadata, distinguish the full checkpoint size from the inference-only exported model size where possible.

Model size should be measured from the saved files; it must not be inferred from parameter count alone.

| Model | Saved checkpoint size | Inference/export model size |
|---|---:|---:|
| RNN | Pending measurement | Pending measurement |
| LSTM | Pending measurement | Pending measurement |
| GRU | Pending measurement | Pending measurement |
| Transformer | Pending measurement | Pending measurement |
| Cross-modal attention model | Pending measurement | Pending measurement |

## 6. Hardware and software environment

Computational results are meaningful only when the hardware and software environment is documented. For each reported experiment, record:

- CPU model and number of available cores.
- GPU model and available memory (VRAM), if present.
- Whether training and inference actually used the GPU or CPU.
- System RAM.
- Operating system and version.
- Deep-learning framework and version.
- Numerical precision (for example, float32 or mixed precision).
- Batch size and relevant input dimensions.

The experiment logs confirm CPU execution for the currently reported runs. A GPU was not used for these measurements. Exact CPU model, physical RAM, and complete hardware specifications are **Pending documentation**.

| Hardware/software detail | Current record |
|---|---|
| Execution device for reported runs | CPU |
| GPU acceleration | Not used in the reported runs |
| Exact CPU model | Pending documentation |
| System RAM | Pending documentation |
| GPU model and VRAM | Pending documentation |
| Operating system/version per run | Pending documentation |
| TensorFlow version per run | Pending documentation |
| Numerical precision | float32 / mixed precision status to be confirmed per final run |

Hardware details must be associated with the relevant experiment. Results collected on different systems should not be presented as a controlled hardware comparison.

## 7. Measurement protocol

To make the computational-cost comparison reproducible:

1. Use the same input dimensions and data-processing pipeline for all models.
2. Record training duration from the start to the end of the training procedure, excluding unrelated setup unless explicitly stated.
3. Report feature extraction/index construction separately from model training and query-time inference.
4. Load each checkpoint before timing inference and report loading time separately.
5. Run warm-up iterations before collecting inference measurements.
6. Measure at least 100 queries where feasible, using the same query set and candidate gallery for each model.
7. Report batch size, mean latency, variability (standard deviation or p95), and throughput.
8. Count parameters from the instantiated architecture and measure checkpoint size from the actual saved files.
9. Document hardware and software settings alongside every measurement.

## 8. Outstanding requirements

The following items remain to be completed before the computational-cost analysis can be considered complete:

- [x] Record preliminary training times for GRU, LSTM, and cross-modal attention configurations.
- [x] Record supplementary GRU learning-rate and attention-ablation training times.
- [ ] Complete the repeated three-seed model comparison and report mean training time and standard deviation.
- [ ] Measure Text-to-Image inference latency and throughput.
- [ ] Measure Image-to-Text inference latency and throughput.
- [ ] Measure total and trainable parameter counts for every final model.
- [ ] Measure saved checkpoint size and, where applicable, inference-only model size.
- [ ] Document exact CPU, GPU, RAM, operating system, framework version, and numerical precision for the final experiments.
- [ ] Measure feature extraction or gallery-index construction time separately if used by the retrieval system.

## 9. Conclusion

The current measurements provide an initial view of training cost for the GRU, LSTM, and cross-modal attention configurations, as well as supplementary learning-rate and ablation experiments. They suggest that the cross-modal attention configuration requires more training time in the measured setup, although the auxiliary matching loss is an experimental confound.

A complete computational-cost comparison cannot yet be concluded. Inference latency, throughput, parameter counts, saved model sizes, full hardware specifications, and repeated-seed statistics remain to be measured or documented. These values will be added when the corresponding experimental results become available.


### 2.4 Phase 5 — GRU data-strategy reference run

A supplementary Phase 5 reference run completed on CPU using GRU, seed 42, Flickr30k fold 1, batch size 1, one epoch, 200 training steps, and 100 validation steps. It used five captions per image, no augmentation, a frozen pretrained ResNet50, AdamW with learning rate 0.0001, and InfoNCE loss (temperature 0.07).

| Configuration | Training time | Training objective | Validation retrieval loss |
|---|---:|---:|---:|
| GRU Phase 5 reference | 7.56 min | 0.819112 | 0.805117 |

This is a separate diagnostic configuration and must not be pooled with the batch-size-4, 200-validation-batch three-model comparison. The checkpoint was reused during resume. The evaluation batch size was increased to 4 while retaining all 5,086 validation images; feature extraction was approximately 1,272 batches rather than 5,086 single-image batches. The one-caption Phase 5 run was still in progress in the latest log, so its metrics and runtime remain Pending.


### 2.1.2 Repeated-seed comparison — current progress

The nine-run comparison is progressing. The following runs have completed under the common fold-1 setup (batch size 4, one epoch, 200 training steps, 200 validation batches, AdamW at 0.0001, InfoNCE temperature 0.07, five captions per image, no augmentation, frozen pretrained ResNet50, CPU execution):

| Model | Seed | Training time | Training objective | Validation retrieval loss |
|---|---:|---:|---:|---:|
| GRU | 42 | 25.16 min | 2.086361 | 1.898083 |
| LSTM | 42 | 22.03 min | 2.089612 | 1.884016 |
| Cross-modal attention (LSTM) | 42 | 55.62 min | 2.235646 | 1.903820 |
| GRU | 123 | 25.51 min | 2.039485 | 1.826844 |

The seed-123 LSTM run is currently in progress (run 5/9); its final training time and metrics are pending. Seed-123 cross-modal attention and all seed-456 runs are pending. The cross-modal attention run includes an auxiliary matching loss (weight 0.2), unlike the GRU/LSTM runs, so its runtime is for the full configuration and does not isolate attention overhead. The held-out test split was not used. Mean and standard deviation will be calculated only after comparable repeated runs are complete.


### 2.5 Phase 5 — one-caption experiment

The GRU one-caption-per-image experiment completed on CPU with seed 42, fold 1, batch size 1, one epoch, 200 training steps, 100 validation batches, AdamW at learning rate 0.0001, InfoNCE temperature 0.07, frozen pretrained ResNet50, and no augmentation.

| Configuration | Training time | Reported training objective | Validation retrieval loss |
|---|---:|---:|---:|
| One caption per image | 7.66 min | 0.000000 | 0.847462 |

**Validity warning:** The exactly zero training objective is anomalous given the non-zero validation retrieval loss. The loss and negative-sampling/batch construction must be inspected before this run is treated as a valid training-cost/performance observation or used to draw conclusions about reducing captions per image. Its runtime is recorded as an observed elapsed time, but the run's optimization result is flagged for verification. The light-augmentation run was in progress in the latest log; its final training time and metrics are pending.
