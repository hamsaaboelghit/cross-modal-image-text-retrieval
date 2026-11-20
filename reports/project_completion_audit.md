# VisionMatch AI — Project Completion Audit

**Repository:** `hamsaaboelghit/cross-modal-image-text-retrieval`  
**Audit scope:** GitHub source, committed documentation/results, and GitHub Actions.  
**Important limitation:** GitHub does not contain the local Flickr30k images, generated vocabulary/encoded data, trained checkpoints, or local experiment logs. Their presence, integrity, and runtime behavior cannot be certified from repository source alone.

## 1. Executive summary

VisionMatch AI has a substantial implementation: dataset preparation, image/text encoders, contrastive objectives, training/evaluation scripts, retrieval gallery construction, FastAPI inference, and a Next.js website.

The backend and website are already present on `main`. A separate draft PR (#5) adds a real backend connection to the desktop application, a website similarity filter, faster dataset file enumeration, and additional setup/validation documentation.

The largest remaining project risks are **experimental evidence and local deployment artifacts**, not missing high-level architecture code.

## 2. Dataset and data pipeline

| Item | Repository evidence | Status |
|---|---|---|
| Flickr30k image/caption preparation | `src/data/prepare_captions.py`, `src/data/prepare_images.py` | Implemented |
| Encoded captions and vocabulary | `src/data/encode_captions.py`, `src/data/build_vocabulary.py`, `src/data/vocabulary.py` | Implemented |
| Image-level train/validation/test splitting | `src/data/split_dataset.py`, committed `data/splits/flickr30k/image_splits.csv` | Implemented |
| Dataset loader | `src/data/flickr30k_dataset.py` | Implemented |
| Caption validation | `src/data/validate_captions.py` | Implemented |
| Image decoding/integrity validation | New `src/data/validate_images.py` in PR #5 | Added; local execution pending |
| Faster image directory scanning | New `src/data/image_files.py` in PR #5 | Added; CI test pending |

Committed split metadata records 31,783 unique images, with 25,426 development images and 6,357 held-out test images. Fold 1 is documented as 20,340 training and 5,086 validation images.

**Not verified:** the actual local image count, caption file count, image decodability, processed CSV/encoded files, vocabulary, and whether the local data matches the committed split file. These require the local dataset.

## 3. Deep-learning models and experiments

### Implemented model families

- CNN retrieval baseline with masked mean-pooled text.
- ResNet50 + RNN.
- ResNet50 + LSTM.
- ResNet50 + GRU.
- ResNet50 + Transformer.
- ResNet50 + LSTM with cross-modal attention and an auxiliary matching branch.

The repository includes InfoNCE, margin-based, triplet, and matching losses, along with experiment runners for architecture comparison, hyperparameter screening, loss comparison, data strategy, and pretrained-versus-scratch initialization.

### Results currently committed

The committed architecture report contains validation metrics for LSTM, GRU, Transformer, and Cross-Modal Attention. It explicitly notes that LSTM came from an earlier run, the new architecture runs used one seed and five epochs, the baseline was not evaluated, and the comparison is not yet controlled. GRU, Transformer, and Cross-Modal Attention results in that report are very low.

Therefore:
- These are preliminary validation results, not final model claims.
- A baseline result is still missing from that comparison.
- A consistent architecture comparison must be rerun with identical settings.
- At least three seeds are still needed for the selected final configurations.
- The held-out test set must remain untouched until model/configuration selection is complete.

### Experimental requirements still outstanding

| Requirement | Current status |
|---|---|
| Baseline + three or more DL approaches | Implemented in code |
| Controlled architecture comparison | Partially evidenced; rerun required |
| Hyperparameter experiments | Scripted; completed results not established |
| Loss-function comparison | Scripted; completed results not established |
| Caption-count/augmentation/negative-sampling experiments | Partially completed in the Phase 5 pilot; see the dated results report below. |
| Pretrained vs. scratch | In progress; the fine-tuning run had started, while the scratch run was still pending at the latest log snapshot. |
| Three-seed final comparison | Not verified |
| Final held-out test evaluation | Not verified |
| Error analysis and failure examples | Not verified |
| Training/inference cost and resource analysis | Not verified |
| Final research conclusions | Not ready | 

### Phase 5 — Data Strategy and Initialization (interim snapshot: 2026-11-20)

The latest user-provided run log confirms two completed GRU pilot variants with validation retrieval-loss values, plus one reused checkpoint. These are **partial pilot results**, not a completed Phase 5 comparison.

| Experiment | Configuration change | Validation retrieval loss | Training time | Status |
|---|---|---:|---:|---|
| `gru_phase5_data_reference` | Five captions; no augmentation; pretrained ResNet50 frozen | Not available in the supplied log excerpt | Reused checkpoint | Evaluated; metric not captured in the excerpt |
| `gru_phase5_one_caption` | One caption per image | 0.847462 | Not captured | Completed |
| `gru_phase5_light_augmentation` | Five captions; light image augmentation | **0.805119** | Not captured | Completed |
| `gru_phase5_resnet50_pretrained` | Five captions; pretrained ResNet50 with last block unfrozen | Pending | Running at the latest log snapshot | In progress |
| `gru_phase5_resnet50_scratch` | Five captions; ResNet50 initialized from scratch with last block unfrozen | Pending | Pending | Not yet run |

Pilot execution settings: one epoch, batch size 1, 200 training batches per experiment, up to 100 validation batches, learning rate 1e-4, AdamW, InfoNCE temperature 0.07, seed 42, Fold 1, and CPU execution on native Windows. The validation dataset contains 5,086 images; the training split contains 20,340 images. The test split was not used. The runner also extracted embeddings for a 5,000-image development gallery after completed/reused runs.

The one-caption run printed a training retrieval loss of 0.0000. With batch size 1, in-batch contrastive negatives are absent, so this training-loss value should not be interpreted as evidence of successful optimization. The validation retrieval loss is retained as the reported metric.

Among the two numerical validation losses captured so far, light augmentation is lower (0.805119 versus 0.847462). This is only a provisional observation: the runs are short, use a single seed, and change data strategy; the final ranking must wait for all variants and consistent full evaluation. No held-out test result or final performance claim is made here.


## 4. Backend

The FastAPI implementation is on `main` and includes:
- `GET /health` and `GET /models`.
- `POST /search/text` and `POST /search/image`.
- `GET /images/{image_name}` and `GET /metrics`.
- Configurable checkpoint, vocabulary, gallery, image directory, CORS, query length, Top-K, and upload size.
- Input validation, upload type/size checks, response schemas, and degraded health behavior when local artifacts are missing.

The backend serves one configured model per API process. Selecting a different model requires another configured process or a future model registry; the request's model field does not hot-swap weights.

**Validation evidence:** the latest committed `main` Backend CI run passed. This is software-contract testing, not proof of successful inference with the user's local checkpoint and gallery.

## 5. Website

The Next.js website on `main` supports:
- Text-to-Image and Image-to-Text modes.
- Image preview and upload validation.
- Model endpoint discovery and Top-K selection.
- Ranked results, similarity values, loading/error/empty states.
- Model Insights populated from saved evaluation artifacts rather than invented scores.

PR #5 adds a minimum-similarity filter to retrieved results and corrects the image-encoder summary label.

**Validation evidence:** the latest committed `main` Frontend CI run passed. PR #5 runs ESLint and a production build against its updated website.

## 6. Desktop GUI

The original PySide6 UI had search validation but did not call the backend.

PR #5 connects it to FastAPI using a background worker and adds:
- Both retrieval directions.
- JPEG/PNG/WEBP upload handling and a 10 MiB client-side limit.
- Top-K selection (5/10/20/50).
- Scrollable ranked results and similarity values.
- Backend/API error messages.
- Optional desktop dependencies and a setup guide.

A desktop smoke-test workflow compiles the desktop package and constructs the UI in offscreen mode. The workflow passed on the PR branch after installing the required Qt runtime libraries.

A real search still depends on a running API configured with a trained checkpoint, matching vocabulary, local images, and a matching gallery.

## 7. GitHub automation and delivery

- Backend CI exists and has passed on `main`.
- Frontend CI exists and has passed on `main`.
- Desktop CI was added in PR #5 and passed on the PR branch.
- PR #5 is a draft targeting `main`; it is intentionally not merged while the remaining audit continues.

## 8. Local artifacts required for a complete demo

The following are intentionally not committed to Git:
- Flickr30k images and original captions.
- Processed captions, encoded captions, and vocabulary.
- Trained TensorFlow checkpoints.
- Retrieval gallery NPZ files.
- Local training logs and final evaluation JSON files.

The application will report degraded readiness or return 503 until the configured model artifacts and gallery are available.

## 9. Final completion checklist

- [x] Main model families and training/evaluation code are present.
- [x] Backend and website are present on main.
- [x] Backend CI and Frontend CI passed on main.
- [x] Desktop backend integration implemented in PR #5.
- [x] Desktop CI and frontend CI passed on PR #5 at the reviewed revision.
- [x] Faster image scanning and explicit image-integrity validation added in PR #5.
- [ ] Confirm local Flickr30k data and run structural/corruption checks.
- [ ] Generate/verify vocabulary and encoded captions for the selected fold.
- [ ] Complete controlled model training and compare baseline + selected models.
- [ ] Complete required multi-seed experiments.
- [ ] Evaluate the selected final model once on the held-out test split.
- [ ] Build the gallery from the exact selected checkpoint and vocabulary.
- [ ] Run an end-to-end live demo through the website and desktop app.
- [ ] Complete error analysis, computational-cost reporting, and final academic report/presentation.

## 10. Conclusion

The software foundation is largely in place. The most important remaining work is to establish trustworthy model results and verify the local inference artifacts. Do not describe the project as experimentally complete until the controlled training, multi-seed validation, final test evaluation, and real end-to-end demo are complete.

## Finalization branch — code-level corrections prepared

Branch: `finalization/fix-retrieval-training`.

The source review identified a concrete cause for the Phase 5 one-caption anomaly: Phase 5 used training batch size 1 with in-batch InfoNCE. A one-image batch has no negative images, so its contrastive objective is degenerate and the logged zero is not evidence of successful learning. The Phase 5 runner is changed on this branch to use batch size 4 and a separate output/checkpoint namespace (`*_batch4`) so the invalid pilot checkpoints are not silently reused. A regression test now requires batch size 1 to fail clearly and a batch of four to produce finite, nonzero loss and gradients.

The projection head now explicitly declares mask handling and returns no sequence mask for pooled rank-2 embeddings; sequence masks are retained only for rank-3 features. This addresses the Keras warning at the projection boundary, but the complete training and inference path still needs to be executed locally to confirm that all warnings and padding-related behavior are resolved.

**Not yet verified:** this branch has not been run on the user's Windows environment. Before merging, run the targeted unit tests, the full test suite, and the new Phase 5 batch-4 pilot. Re-evaluate the resulting checkpoint with the exact same architecture configuration, rebuild the gallery from that checkpoint and vocabulary, then smoke-test both API retrieval directions. If retrieval metrics remain weak, report that honestly; these changes prevent a degenerate training setup but do not guarantee high retrieval quality.

The previous Phase 5 batch-1 results remain historical pilot results and must not be relabeled as corrected experiments. The test split remains unused.
