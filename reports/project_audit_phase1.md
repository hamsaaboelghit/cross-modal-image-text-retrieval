# Phase 1 — Project Implementation Audit

**Repository:** `hamsaaboelghit/cross-modal-image-text-retrieval`  
**Branch reviewed:** `main`  
**Reviewed base commit:** `bcfe72f4ddfecdb30f014aa6a4b01aa19f34744f`  
**Audit scope:** repository code and documentation visible on GitHub. Local-only datasets, checkpoints, logs, and runtime behavior are not verified by this audit.

## 1. Executive summary

The repository already contains substantial research and backend implementation. It should not be rebuilt from scratch.

The main remaining risks are:
- the five architectures have experiment definitions, but completed training runs and comparable validation results have not yet been verified;
- the current audit identifies fairness differences between the baseline and full models;
- hyperparameter experiment code accepts an embedding dimension, but the full-model path currently fixes its retrieval output dimension at 512;
- the inference model construction currently sets `pretrained=False`, while the research experiment configuration documents pretrained ResNet50;
- the documented project status says user-facing website integration remains future work.

This report distinguishes implementation evidence from completed experimental evidence. A script or configuration is not proof that a run completed.

## 2. Person One — Deep Learning & Model Experiments

| Requirement | Audit status | Evidence / issue |
|---|---|---|
| Baseline CNN + Mean Pooling | Implemented in code | Model architecture and experiment configuration are documented in the existing model audit. |
| ResNet50 + LSTM | Implemented in code | Experiment configuration is documented; completed run still needs artifact verification. |
| ResNet50 + GRU | Implemented in code | Experiment configuration is documented; completed run still needs artifact verification. |
| ResNet50 + Transformer | Implemented in code | Experiment configuration is documented; completed run still needs artifact verification. |
| ResNet50 + Cross-Modal Attention | Implemented in code | Dedicated matching branch/loss is an additional objective and affects strict comparability. |
| Common dataset split | Partially verified | The experiment runner loads train and validation for the configured fold. Actual split identity across all historical runs needs logs/config artifacts. |
| Common preprocessing | Needs verification | ResNet50 applies its own preprocessing internally; baseline and full-model image paths require an explicit equivalence review. |
| Common caption preprocessing | Needs verification | Backend tokenization comments claim parity with training tokenization; tests should verify exact parity. |
| Shared embedding dimension | Partially implemented | Runner passes `output_dim` to the baseline, but the full-model branch does not pass it and therefore remains at its model default (512). |
| Documented loss, batch size, LR, epochs, seed | Partially implemented | Runner configures these values; per-run artifacts and all architecture-specific settings still need validation. |
| Consistent stopping policy | Partially implemented | Runner configures early stopping; confirm that all architecture scripts use the same policy and that logs record actual epochs. |
| Named checkpoints | Implemented in configuration | Checkpoint directories are configured per experiment; actual checkpoint files are not part of this repository audit. |
| Validation-only model selection | Supported by design | The experiment runner loads train and validation; the hyperparameter evaluator records validation selection and states that test is not used. Verify actual run artifacts. |
| Hyperparameter experiments | Scripted; results unverified | An OFAT experiment runner and plotting utility exist. No claim of completed training or metrics is made here. |
| Loss comparison | Implementations exist; comparison unverified | InfoNCE, margin-based, and triplet loss modules exist. A controlled training/evaluation comparison remains to be demonstrated. |
| Data strategy experiments | Not verified | No complete evidence of the requested augmentation, caption-count, and negative-sampling comparisons was established in this audit. |
| Pretrained vs. scratch | Not verified | The runner supports pretrained selection, but a completed controlled comparison has not been established. |
| Three-seed final training | Not verified | No verified three-seed final result bundle was established. |
| Research report | Partially documented | `reports/model_comparison_audit.md` exists; final experimental results and discussion remain pending. |

### 2.1 Fairness issues already identified

The existing `reports/model_comparison_audit.md` identifies:
1. Baseline and full models use different image encoders (custom CNN versus ResNet50).
2. Default dropout differs (0.1 baseline versus 0.2 full models).
3. Cross-Modal Attention includes an additional matching-loss component.
4. Model capacity differs and should be reported.
5. The experiment runner's `output_dim` parameter affects the baseline path but is not passed to the full-model constructor.

These differences do not automatically invalidate the architecture study, but they must be controlled where possible or explicitly disclosed.

## 3. Person Three — Backend & Website

| Requirement | Audit status | Evidence / issue |
|---|---|---|
| FastAPI application | Implemented | `app/backend/main.py` contains the application and routes. |
| Health endpoint | Implemented | `GET /health` reports backend status and whether the retrieval service is loaded. |
| Text-to-Image endpoint | Implemented | `POST /search/text` calls the retrieval service. |
| Image-to-Text endpoint | Implemented | `POST /search/image` accepts an uploaded image and returns captions. |
| Model loading/service | Implemented | `app/backend/model_service.py` builds the selected architecture and restores a TensorFlow checkpoint. |
| Retrieval service | Implemented | Backend model/retrieval service code is present. |
| Input/error handling | Partial | Empty text and invalid image handling are present. A strict upload-size limit and comprehensive content-type validation were not established from the reviewed route. |
| `GET /models` | Not found in reviewed route file | Add an endpoint that lists supported models and clearly indicates availability. |
| Model selection | Partial | Model type is configured at service initialization; per-request selection and a user-facing model list are not established. |
| Checkpoint portability | Needs improvement/verification | Checkpoint discovery uses configured local paths. Document a portable artifact layout and environment-based configuration. |
| Pretrained/inference compatibility | High-priority verification | The reviewed inference builder passes `pretrained=False`; the experiment audit documents pretrained ResNet50 for full-model research runs. Confirm architecture/weight compatibility before serving a checkpoint. |
| Swagger/OpenAPI | Available through FastAPI | Confirm request/response schemas and examples are complete. |
| Automated backend tests | Not fully verified | A complete route and integration test suite was not established by the files reviewed in this phase. |
| User-facing website | Not verified as complete | README lists integration with a user-facing website/application as further research and development. |
| Model comparison page | Not verified | Must use actual stored evaluation results rather than manually entered metrics. |

### 3.1 Backend files reviewed

- `app/backend/main.py`
- `app/backend/model_service.py`

The reviewed `main.py` contains root, health, text-search, and image-search routes. The reviewed model service loads a vocabulary, builds a model, resolves a checkpoint, and restores weights.

## 4. Priority order for subsequent work

| Priority | Work item | Why it comes next |
|---|---|---|
| P0 | Verify local checkpoints, logs, and dataset availability | Avoid rerunning completed work and avoid claiming unverified results. |
| P0 | Resolve inference architecture/checkpoint compatibility | A mismatch can prevent the deployed model from restoring or reproducing training behavior. |
| P1 | Fix and test embedding-dimension propagation | Required for valid embedding-dimension experiments. |
| P1 | Establish a controlled architecture comparison protocol | Make preprocessing, split, training budget, and evaluation comparable. |
| P1 | Complete hyperparameter and loss comparisons | Produce real validation metrics and logs. |
| P2 | Complete data-strategy and pretrained-vs-scratch experiments | Cover remaining research requirements. |
| P2 | Run final configurations with at least three seeds | Quantify variability and produce final checkpoints. |
| P1 | Complete backend validation, model listing, and upload safeguards | Make inference reliable and safer to expose. |
| P2 | Build and connect the user-facing website | Complete the user-facing product. |
| P3 | End-to-end tests and final documentation | Verify the integrated system and prepare delivery. |

## 5. Evidence and limitations

This audit is based on repository files and documentation available on the reviewed GitHub branch. It does not inspect the user's local machine, hidden/ignored files, runtime process state, dataset contents, or checkpoint binaries. Therefore:

- "Implemented" means relevant code was found, not that every runtime path has passed testing.
- "Not verified" does not necessarily mean the work does not exist locally.
- No experimental metric, training completion, or checkpoint is claimed without a verifiable artifact.
- The held-out test split must remain excluded from hyperparameter and model selection.

## 6. Files for the next phase

The next implementation phase should begin with local artifact verification and a focused review of:
- `src/training/experiment_runner.py`
- `src/training/run_hyperparameter_experiments.py`
- `src/models/full_model.py`
- `app/backend/main.py`
- `app/backend/model_service.py`
- `reports/model_comparison_audit.md`

**Phase 1 outcome:** initial repository audit documented. No model code, training configuration, dataset, checkpoint, or backend behavior was changed as part of this audit.
