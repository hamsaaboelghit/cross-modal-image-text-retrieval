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

