# VisionMatch AI - Architecture Comparison

## Experimental setup
- Dataset: Flickr30k
- Split: Fold 1 validation
- Validation images: 5,086
- Captions per image: 5
- Embedding dimension: 512
- Test split: Not used
- New architecture experiments used one seed and five epochs.

## Retrieval results

| Model | Direction | R@1 | R@5 | R@10 | Median Rank | MRR |
|---|---|---:|---:|---:|---:|---:|
| LSTM | Text-to-Image | 3.4959% | 18.6394% | 28.4978% | 33 | 0.118431 |
| LSTM | Image-to-Text | 5.2890% | 23.7711% | 34.7031% | 24 | 0.148898 |
| GRU | Text-to-Image | 0.0118% | 0.1062% | 0.1848% | 2534 | 0.001728 |
| GRU | Image-to-Text | 0.0197% | 0.0983% | 0.2556% | 3633 | 0.001660 |
| Transformer | Text-to-Image | 0.0118% | 0.0786% | 0.2045% | 2530 | 0.001696 |
| Transformer | Image-to-Text | 0.0197% | 0.0983% | 0.1770% | 3697 | 0.001643 |
| Cross-Modal Attention | Text-to-Image | 0.0236% | 0.1770% | 0.3854% | 1772 | 0.002866 |
| Cross-Modal Attention | Image-to-Text | 0.0197% | 0.2359% | 0.4129% | 3723 | 0.002613 |

## Preliminary observations
- LSTM has the strongest retrieval metrics in both directions.
- GRU and Transformer have very weak retrieval performance in these runs.
- Cross-Modal Attention improves several metrics over GRU and Transformer, but remains below LSTM.
- Cross-Modal Attention has the lowest validation loss, but not the best retrieval performance.

## Limitations
- LSTM comes from an earlier training run, so this is not yet a fully controlled comparison.
- New architecture experiments used one seed and five epochs.
- The required simple baseline has not yet been evaluated.
- No final conclusion until training conditions are aligned and multiple seeds are evaluated.
- Test split remains untouched.

## Next actions
1. Add and evaluate the simple baseline.
2. Align training conditions for a fair architecture comparison.
3. Conduct hyperparameter and data-strategy experiments.
4. Repeat selected final configurations with at least three seeds.
5. Use the test split for final evaluation only.




## Latest fair-comparison progress — 20 November 2026

The earlier table above contains historical architecture retrieval metrics from runs with different training histories; it must not be treated as the final controlled comparison.

### Newly completed run: LSTM, fold 1, seed 123

- Run: `fair200_20261120_lstm_fold1_seed123`
- Configuration: batch size 4, one epoch, 200 training steps, 200 validation steps, AdamW, learning rate 0.0001, weight decay 0.0001, InfoNCE temperature 0.07, pretrained frozen ResNet50, five captions per image.
- Device: CPU.
- Training objective/retrieval loss: **2.060249**.
- Validation retrieval loss: **1.856892**.
- Training time: **22.93 minutes**.
- Best checkpoint: `checkpoints/experiments/fair200_20261120_lstm_fold1_seed123/best/best_model-1`.
- Test split: not used.

This log does not include Recall@K, Mean Recall, MRR, or Median Rank. Therefore, this run cannot yet be added to the retrieval-results table above.

### In-progress run: Cross-Modal Attention, fold 1, seed 123

Run `fair200_20261120_cross_attention_fold1_seed123` (run 6/9) has started. It uses an LSTM text encoder, Cross-Modal Attention, and a matching branch with matching-loss weight 0.2. The run is configured for one epoch, batch size 4, 200 training steps and 200 validation steps, AdamW, learning rate 0.0001, InfoNCE temperature 0.07, and CPU execution.

No final metrics or checkpoint have been reported yet. Training/validation losses and retrieval-ranking metrics remain **Pending** until completion.

### Notes

- The Keras mask warnings did not prevent the LSTM run from completing, but they should be investigated before final reporting.
- Validation retrieval loss is not interchangeable with Recall@K or Mean Recall.
- No final architecture winner can be declared from this partial run update.
- The held-out test split remains unused.


---

## Updated status and research protocol — 20 November 2026

### Explicit hypotheses

These are proposed hypotheses to be tested, not conclusions:

- **H1:** Recurrent text encoders will outperform the lightweight CNN/mean-pooling baseline on bidirectional retrieval when evaluated under matched conditions.
- **H2:** GRU and LSTM will exhibit different retrieval-quality/computational-cost trade-offs.
- **H3:** Cross-modal attention will improve retrieval only if it is compared with a recurrent control using the same loss terms and training budget.
- **H4:** Caption count and image augmentation will affect retrieval generalization and must be tested as separate data-strategy factors.
- **H5:** Pretrained ResNet50 will provide a more effective or efficient visual representation than scratch initialization on this dataset.

### Latest run correction

The latest received log confirms that seed-123 LSTM completed; it supersedes the earlier statement in this report that this run was still in progress.

| Run | Status | Training objective | Validation retrieval loss | Training time |
|---|---|---:|---:|---:|
| GRU, seed 42 | Completed | 2.086361 | 1.898083 | 25.16 min |
| LSTM, seed 42 | Completed | 2.089612 | 1.884016 | 22.03 min |
| Cross-Modal Attention, seed 42 | Completed | 2.235646 | 1.903820 | 55.62 min |
| GRU, seed 123 | Completed | 2.039485 | 1.826844 | 25.51 min |
| LSTM, seed 123 | Completed | 2.060249 | 1.856892 | 22.93 min |
| Cross-Modal Attention, seed 123 | In progress in latest supplied log | Pending | Pending | Pending |
| GRU, seed 456 | Pending | Pending | Pending | Pending |
| LSTM, seed 456 | Pending | Pending | Pending | Pending |
| Cross-Modal Attention, seed 456 | Pending | Pending | Pending | Pending |

All completed runs above used fold 1, CPU, batch size 4, one epoch, 200 training steps, and 200 validation steps. The seed-123 LSTM run used AdamW (learning rate 0.0001, weight decay 0.0001), InfoNCE temperature 0.07, five captions per image, and frozen pretrained ResNet50. Its best checkpoint is `checkpoints/experiments/fair200_20261120_lstm_fold1_seed123/best/best_model-1`. The test split was not used.

**Important:** These are objective/loss and runtime values, not retrieval-ranking results. The table above cannot establish the best retrieval architecture. The Cross-Modal Attention setup uses an auxiliary matching loss with weight 0.2, unlike the GRU/LSTM runs; this confounds an attention-only comparison. The Keras sequence-mask warnings also need investigation.

### Required next analyses

1. Complete the pending runs and evaluate every checkpoint with the same validation retrieval evaluator.
2. Report T2I and I2T Recall@1/5/10, Mean Recall, MRR, and Median Rank for every seed; calculate mean ± standard deviation.
3. Add baseline and Transformer to the same controlled comparison if making claims across all implemented architecture families, or narrow the report's claim.
4. Isolate cross-modal attention and matching loss in separate ablations.
5. Verify the one-caption experiment whose training objective was reported as exactly zero.
6. Complete augmentation and pretrained-vs-scratch/fine-tuning studies.
7. Replace the error-analysis placeholders with actual query/top-five examples and documented counts.
8. Measure parameter counts, checkpoint size, inference latency/throughput, memory, and exact hardware.
9. Select using validation only, then evaluate the held-out test split once.
10. Run and record the final demo using the selected checkpoint and its matching gallery.

### Error-analysis protocol

For both text-to-image and image-to-text, save a reproducible sample of queries, ground-truth matches, top-five results, ranks/similarity scores, and human-reviewed error labels: object confusion, action confusion, attribute mismatch, spatial-relation mismatch, multiple-object omission, and semantic mismatch. Counts and examples remain pending until actual predictions are inspected.

### Reporting metrics and query groups

Because this is a ranking task, ordinary per-class classification accuracy is not directly applicable. If category labels exist, report retrieval metrics by category; otherwise define reproducible query groups (e.g., people, animals, actions, attributes, spatial relations), provide group sizes, and report group-wise retrieval metrics. Do not invent categories or examples.
