# VisionMatch AI — Technical Report

**Project title:** VisionMatch AI: Cross-Modal Image–Text Retrieval Using Deep Learning  
**Team:** Hamsa Mustafa Aboelghit, Toqaa Mohamed Hassan, Mariam Mohamed Abdelkawey, Yasmen Abdelaziz Ramadan, Basma Sameh Mohamed  
**Document status:** Working draft — evidence-led; incomplete results are explicitly marked **Pending**.  
**Dataset:** Flickr30k, fold 1  
**Evaluation policy:** Validation is used for model selection. The held-out test split remains reserved for the final evaluation after model selection.  
**Target final length:** 15–25 pages after adding complete results, figures, error examples, and formatted references.

> **Reporting rule.** Results from different training budgets, seeds, batch sizes, or loss configurations are not pooled as though they were a controlled comparison. Historical/diagnostic results are labelled separately. No final best model is declared until comparable retrieval metrics and repeated-seed results are available.

---

## 1. Abstract

VisionMatch AI is a bidirectional cross-modal retrieval system that learns a shared semantic representation space for images and natural-language captions. It supports two tasks: Text-to-Image (T2I) retrieval, in which a text query is used to rank relevant images, and Image-to-Text (I2T) retrieval, in which an input image is used to rank its associated captions. The project uses the Flickr30k dataset, which contains 31,783 images paired with 158,915 human-written captions.

The system follows a dual-encoder design. An image encoder extracts visual representations, while a text encoder represents caption sequences. Projection layers map both modalities into a shared 512-dimensional embedding space. The project investigates a lightweight CNN/mean-pooling baseline and multiple sequence encoders, including RNN, LSTM, GRU, and Transformer variants. An advanced cross-modal attention branch is also implemented for image–text matching. Contrastive learning with an InfoNCE objective is used to align paired image and text representations.

The evaluation protocol reports retrieval-ranking metrics in both directions, including Recall@1, Recall@5, Recall@10, Mean Recall, Mean Reciprocal Rank (MRR), and Median Rank. Preliminary experiments demonstrate that validation loss alone is not a reliable indicator of retrieval quality; consequently, model selection and reporting prioritize task-appropriate retrieval metrics. The controlled multi-seed comparison, final held-out test results, complete ablations, error analysis, and several computational-cost measurements are **Pending**. Therefore, this report does not claim a final winning architecture.

## 2. Introduction

The ability to associate visual content with natural language is a central problem in multimodal machine learning. Image–text retrieval addresses this problem by ranking images or captions according to their semantic compatibility rather than generating a new caption. This ranking formulation supports practical search applications: a user can describe a scene in natural language and retrieve matching images, or provide an image and retrieve descriptions that refer to it [1, 2].

VisionMatch AI studies bidirectional retrieval on Flickr30k. Each image is associated with five human-written descriptions, allowing the model to learn from multiple linguistic views of the same visual scene. The project adopts a shared embedding approach: image and text encoders independently produce feature vectors, which are projected and normalized into a common space. Similarity between image and text embeddings is then used to rank candidates.

The image branch uses a ResNet50-based encoder in the modern model family. Residual connections were introduced to facilitate the optimization of deeper neural networks [3]. The text branch is varied across recurrent and attention-based architectures. RNNs provide a sequential baseline; LSTM and GRU introduce gating mechanisms to model dependencies across token sequences [4, 5]; and the Transformer uses attention rather than recurrence to build contextual representations [6].

The project is structured as a comparative study rather than a single-model demonstration. It includes a lightweight baseline, multiple text-encoder families, a cross-modal matching branch, controlled training experiments, data-strategy experiments, and ablation studies. Since retrieval is a ranking task, accuracy is not used as the default metric. Both retrieval directions are evaluated independently, and the final model will be selected using validation retrieval performance under comparable experimental conditions.

## 3. Problem Statement

Given a collection of images and their associated natural-language captions, the task is to learn a scoring function that ranks semantically relevant items above irrelevant candidates.

Let the image collection be \(I = \{i_1, i_2, \ldots, i_N\}\), and let the set of captions be \(T = \{t_1, t_2, \ldots, t_M\}\). Each image in Flickr30k has five reference captions. The model learns an image encoder \(f_\theta\) and a text encoder \(g_\phi\), followed by projection functions into a shared embedding space:

\[
z_i = \operatorname{norm}(W_i f_\theta(i)), \qquad
z_t = \operatorname{norm}(W_t g_\phi(t)).
\]

The similarity between an image and a caption is measured using their normalized embedding similarity, implemented as a dot product/cosine similarity.

**Text-to-Image (T2I):** for a text query \(t\), rank all candidate images by similarity \(s(t,i)\), with the source image treated as relevant.

**Image-to-Text (I2T):** for an image query \(i\), rank candidate captions by similarity \(s(i,t)\), treating all five captions belonging to that image as relevant.

The system must learn semantic alignment despite differences between visual features and linguistic representations. It must also generalize to unseen validation and test images without allowing captions from the same image to leak across data splits.

## 4. Motivation

A cross-modal retrieval system provides a natural interface for searching visual collections without requiring users to know filenames, tags, or fixed object categories. Natural-language queries can express objects, attributes, actions, and scene relationships. Conversely, image-to-text retrieval can surface human-written descriptions that help users understand or index visual content.

The project is motivated by four technical questions:

1. How do different text sequence encoders (RNN, LSTM, GRU, and Transformer) affect bidirectional image–text retrieval when paired with a common visual encoder?
2. Does a more complex cross-modal attention/matching branch improve retrieval or only the separate image–text matching objective?
3. How do data choices, including the number of captions per image and light image augmentation, affect retrieval quality?
4. What is the trade-off between retrieval quality and computational cost, including training time, inference latency, parameter count, and model size?

The project also emphasizes reproducibility. Architecture comparisons must use the same data split, preprocessing, training budget, and evaluation protocol wherever a fair comparison is claimed. Multiple random seeds are required for the final comparison.

## 5. Related Work

### 5.1 Image–text retrieval as a ranking task

Hodosh, Young, and Hockenmaier framed image description as a ranking problem and studied datasets and evaluation metrics for image–sentence retrieval [1]. This formulation is directly relevant to VisionMatch AI: the objective is to rank relevant image–caption pairs, not to generate a caption token by token.

### 5.2 Flickr30k

Young et al. introduced the Flickr30k corpus and its denotation-graph resource, containing 31,783 images and 158,915 captions [2]. The multiple captions per image make the dataset suitable for studying semantic alignment between visual content and varied natural-language descriptions.

### 5.3 Visual feature extraction with ResNet

ResNet introduced residual learning to make deeper convolutional networks easier to optimize [3]. VisionMatch AI uses ResNet50 as the pretrained visual backbone in the modern model family. The baseline uses a smaller CNN trained as part of the retrieval model, providing a simpler visual encoder for comparison.

### 5.4 Recurrent text encoders

RNNs process token sequences through recurrent hidden states. LSTM addresses long-range dependency learning through gated memory cells [4]. The GRU was introduced as a gated recurrent architecture in the RNN Encoder–Decoder work [5]. VisionMatch AI compares these sequence encoders under a shared embedding and retrieval objective.

### 5.5 Transformer encoders

The Transformer architecture relies on attention mechanisms and removes recurrence from the sequence-transduction architecture [6]. The project includes a Transformer text-encoder variant to compare attention-based contextualization with recurrent sequence processing.

### 5.6 Contrastive multimodal learning

Contrastive learning brings representations of related items closer while separating unrelated examples. CLIP demonstrated large-scale image–text representation learning from paired image and text supervision [7]. VisionMatch AI uses a project-scale contrastive setup with Flickr30k rather than claiming to reproduce CLIP's scale or pretrained model. The implemented training objective is InfoNCE-based; the exact sampling and loss configuration is recorded with each experiment.

## 6. Dataset

The primary dataset is Flickr30k. The official dataset page states that it contains 31,783 Flickr images and 158,915 crowd-sourced captions, with five captions per image [2].

### Table 1 — Dataset comparison

| Dataset | Source | Size | Format | Classes | Licence / usage | Role |
|---|---|---:|---|---|---|---|
| Flickr30k | Official University of Illinois dataset page [2] | 31,783 images; 158,915 captions; 5 captions/image | Image files plus caption annotations; project preprocessing stores encoded captions in CSV/token-sequence form | Not a fixed-class classification dataset; semantic relevance is defined by image–caption pairing | Image copyright remains with original owners. Official page permits research/educational non-commercial use under Flickr Terms of Use; users must comply with source terms | Train / validation / held-out test, using image-level fold 1 split |

The project uses fold 1. The current experiment logs report 20,340 training images and 5,086 validation images. The held-out test split is reserved for final evaluation and has not been used for model selection in the reported runs.

**Dataset limitations:** Flickr30k is relatively small compared with large-scale web-paired datasets. Captions may omit objects or details visible in the image, and multiple valid descriptions may express the same scene differently. Therefore, a single caption should not be treated as the only possible semantic description of an image.

## 7. Data Preprocessing

### 7.1 Image preprocessing

- Images are loaded as RGB.
- Images are resized to 224 × 224 pixels using bilinear interpolation.
- The data pipeline produces float32 image tensors in the range [0, 255].
- The ResNet image encoder applies its required preprocessing internally.
- The baseline CNN scales image values to [0, 1] inside its image encoder.

### 7.2 Caption preprocessing

- Captions are represented as integer token sequences.
- The configured maximum sequence length is 34 tokens.
- Padding uses token ID 0.
- The vocabulary is generated for the corresponding data fold and loaded by the model.
- The standard setup retains all five captions per image. A separate Phase 5 experiment selects one caption per image using a fixed random seed.

### 7.3 Data splitting and leakage prevention

The dataset is split at the image level. All captions associated with one image remain in the same split. This prevents captions for an image from appearing in training while the same image appears in validation or test.

Validation data is used for model selection. The test split is not used during architecture selection, hyperparameter tuning, or ablation decisions.

### 7.4 Augmentation

The standard configuration uses no image augmentation. A separate light-augmentation experiment applies stateless random brightness, contrast, and saturation transformations to training images only. Validation and test images are not augmented.

## 8. Methodology

VisionMatch AI uses a dual-encoder retrieval pipeline:

1. An image is passed through the visual encoder to obtain a visual feature representation.
2. A caption is tokenized and passed through the text encoder to obtain a sequence representation.
3. Modality-specific projection layers map the image and text features into a shared 512-dimensional embedding space.
4. Embeddings are L2-normalized.
5. Similarity scores are computed between query and candidate embeddings.
6. Candidates are sorted by similarity to produce a ranked retrieval list.

For an image–caption pair, the training objective encourages matching representations to have higher similarity than non-matching pairs. The main training loss is InfoNCE with temperature 0.07 in the documented experiments. Random negative sampling is used in the current Phase 5 study.

The system is evaluated in both directions. For T2I, each of the five captions is treated as a query whose corresponding image is the positive candidate. For I2T, the source image is the query and all five captions associated with that image are treated as relevant candidates.

The advanced cross-modal attention branch additionally combines token-level text features and spatial image features to compute an image–caption matching score. In the current implementation, the standard retrieval evaluator uses the independent image and text embeddings; it does not use the cross-modal matching score to rank candidates. Consequently, retrieval metrics and matching-branch metrics must be reported separately. This distinction is essential when interpreting the contribution of cross-modal attention.

## 9. Baseline

The baseline is a lightweight dual-encoder architecture designed to provide a simpler reference point than the pretrained ResNet50 and sequence-model variants.

**Image encoder:** four convolutional stages with 32, 64, 128, and 256 channels. Each stage uses a 3 × 3 convolution, batch normalization, and ReLU activation. Global average pooling produces a fixed-size visual representation, followed by a projection to the shared embedding dimension.

**Text encoder:** a trainable token embedding layer, masked mean pooling over non-padding tokens, dropout, and a dense projection to the shared embedding dimension.

Both image and text embeddings are L2-normalized before similarity-based retrieval. The baseline does not use a pretrained ResNet backbone or a recurrent text encoder.

**Baseline final retrieval metrics:** Pending. The previously recorded baseline training run and its validation loss are not sufficient to fill the retrieval-ranking result table.

## 10. Proposed/Advanced Models

### 10.1 ResNet50 + RNN

The image branch uses a pretrained ResNet50 visual encoder. The text branch uses a recurrent sequence encoder. This model provides a recurrent reference for comparing gated and attention-based text encoders.

### 10.2 ResNet50 + LSTM

The LSTM text encoder uses gated memory cells to model token dependencies. Its output is projected into the shared 512-dimensional embedding space. The architecture is motivated by the ability of LSTM networks to preserve information across longer sequence intervals [4].

### 10.3 ResNet50 + GRU

The GRU text encoder uses update and reset gates to summarize caption sequences. It provides a gated recurrent alternative to LSTM [5].

### 10.4 ResNet50 + Transformer

The Transformer text encoder uses attention-based sequence modeling rather than recurrent state transitions [6]. Its retrieval performance and computational cost are compared with the recurrent variants.

### 10.5 Cross-Modal Attention

The advanced model adds a cross-modal attention and matching branch. It uses spatial image features and token-level text features, with an eight-head cross-modal attention configuration in the model implementation. A matching head predicts compatibility for paired and sampled negative image–caption examples.

**Important implementation detail:** The current retrieval evaluator ranks using the independent image/text embeddings, while the cross-modal branch produces separate matching scores. The report will not claim that this branch improves T2I/I2T retrieval unless an experiment explicitly demonstrates that result. Matching performance will be reported separately.

### Table 2 — Model architecture comparison

| Model | Family | Input | Main layers | Parameters | Pretrained? |
|---|---|---|---|---|---|
| Baseline | Lightweight dual encoder | RGB image 224×224×3; token IDs, length 34 | 4-stage CNN + global average pooling; token embedding + masked mean pooling; projection heads | Pending measurement | No |
| ResNet50 + RNN | CNN + recurrent | Same image/text input | ResNet50; RNN text encoder; modality projections | Pending measurement | ResNet50 pretrained in modern configuration |
| ResNet50 + LSTM | CNN + recurrent | Same image/text input | ResNet50; LSTM text encoder; modality projections | Pending measurement | ResNet50 pretrained in modern configuration |
| ResNet50 + GRU | CNN + recurrent | Same image/text input | ResNet50; GRU text encoder; modality projections | Pending measurement | ResNet50 pretrained in modern configuration |
| ResNet50 + Transformer | CNN + Transformer | Same image/text input | ResNet50; Transformer text encoder; modality projections | Pending measurement | ResNet50 pretrained in modern configuration |
| ResNet50 + LSTM + Cross-Modal Attention | CNN + recurrent + cross-modal attention | Image plus caption token sequence and spatial image features | ResNet50; LSTM; 8-head cross-modal attention; matching head; modality projections | Pending measurement | ResNet50 pretrained in documented configuration |

Parameter counts are intentionally left pending until measured from the actual instantiated models. Total and trainable parameter counts must be distinguished, especially when ResNet50 is frozen or partially fine-tuned.

## 11. Experimental Setup

Experiments are organized into separate groups because they use different compute budgets and answer different research questions.

### 11.1 Dataset and split

- Dataset: Flickr30k.
- Fold: 1.
- Training images: 20,340.
- Validation images: 5,086.
- Test split: held out; not used in the reported preliminary runs.

### 11.2 Preliminary three-model comparison

A preliminary comparison was run for GRU, LSTM, and Cross-Modal Attention using seed 42, one epoch, 200 training steps, 200 validation batches, and batch size 4. The image encoder was pretrained ResNet50 and frozen. The optimizer was AdamW, learning rate 0.0001, and the retrieval loss was InfoNCE with temperature 0.07.

The cross-modal attention configuration also enabled an auxiliary matching loss with weight 0.2, unlike the GRU and LSTM runs. The result is therefore a comparison of complete configurations, not an isolated attention-only comparison.

### 11.2.1 Additional completed run — seed 123 GRU

The next run in the repeated-seed comparison completed for the GRU configuration using seed 123. It used the same fold-1 setup as the seed-42 comparison: one epoch, 200 training steps, 200 validation batches, batch size 4, AdamW, learning rate 0.0001, InfoNCE temperature 0.07, pretrained frozen ResNet50, five captions per image, and no image augmentation.

| Model | Seed | Training objective | Validation retrieval loss | Training time |
|---|---:|---:|---:|---:|
| GRU | 123 | 2.039485 | 1.826844 | 25.51 min |

The seed-123 GRU validation retrieval loss is lower than the seed-42 GRU result (1.898083), illustrating why repeated seeds are required. This single additional run does not establish a model ranking. At the time of this update, seed-123 LSTM and cross-modal attention, and all seed-456 runs, remain **Pending**. The held-out test split was not used.

### 11.3 Phase 5 data and initialization studies

A separate GRU study uses seed 42, fold 1, batch size 1, 200 training steps, 100 validation steps, one epoch, learning rate 0.0001, and pretrained ResNet50 unless the experiment explicitly tests scratch initialization. It contains five configurations:

1. Reference: five captions, no augmentation, pretrained frozen ResNet50.
2. One-caption ablation: one caption per image.
3. Light augmentation: five captions and light image augmentation.
4. Pretrained fine-tuning: pretrained ResNet50 with its last block unfrozen.
5. Scratch initialization: randomly initialized ResNet50 with its last block unfrozen.

The current script uses evaluation batch size 4 for full validation retrieval evaluation. The training batch size remains 1. These runs are supplementary studies and must not be merged with the three-seed architecture comparison.

### 11.4 Repeated seeds

The final controlled comparison requires seeds 42, 123, and 456. Results must be reported per seed and as mean ± standard deviation. Until all required runs complete under comparable conditions, the final three-seed table remains Pending.

## 12. Hyperparameters

### Table 3 — Hyperparameter configuration

| Experiment/configuration | Optimizer | Learning rate | Training batch size | Epochs / steps | Regularization | Loss | Augmentation |
|---|---|---:|---:|---|---|---|---|
| Preliminary GRU/LSTM/Cross-Attention comparison | AdamW | 1e-4 | 4 | 1 epoch; 200 train steps; 200 validation batches | Weight decay 1e-4; dropout 0.1 | InfoNCE, temperature 0.07; Cross-Attention also used matching loss weight 0.2 | None |
| Phase 5 GRU reference | AdamW | 1e-4 | 1 | 1 epoch; 200 train steps; 100 validation steps | Weight decay 1e-4; dropout 0.1; gradient clipping 1.0 | InfoNCE, temperature 0.07 | None |
| Phase 5 one-caption | AdamW | 1e-4 | 1 | 1 epoch; 200 train steps; 100 validation steps | Same as Phase 5 reference | InfoNCE, temperature 0.07 | None |
| Phase 5 light augmentation | AdamW | 1e-4 | 1 | 1 epoch; 200 train steps; 100 validation steps | Same as Phase 5 reference | InfoNCE, temperature 0.07 | Light brightness/contrast/saturation |
| Phase 5 pretrained fine-tuning | AdamW | 1e-4 | 1 | 1 epoch; 200 train steps; 100 validation steps | Same; final ResNet block unfrozen | InfoNCE, temperature 0.07 | None |
| Phase 5 scratch initialization | AdamW | 1e-4 | 1 | 1 epoch; 200 train steps; 100 validation steps | Same; ResNet50 randomly initialized and final block trainable | InfoNCE, temperature 0.07 | None |

The Phase 5 reference run reported a training objective of 0.819112, validation retrieval loss of 0.805117, and training duration of 7.56 minutes. This is a limited-budget run with batch size 1 and 200 training steps; it must not be compared directly with the preliminary batch-size-4 architecture comparison as if the training conditions were identical.

## 13. Results

### 13.1 Preliminary three-model comparison (seed 42)

| Model | Training objective | Validation retrieval loss | Training time |
|---|---:|---:|---:|
| GRU | 2.086361 | 1.898083 | 25.16 min |
| LSTM | 2.089612 | **1.884016** | 22.03 min |
| Cross-Modal Attention (LSTM) | 2.235646 | 1.903820 | 55.62 min |

The LSTM has the lowest validation retrieval loss among these three preliminary runs. This is not a final model selection result: only one seed is represented, the training budget is limited, and the Cross-Modal Attention run includes an additional matching-loss objective.

### 13.2 Historical retrieval evaluation

An earlier architecture evaluation produced the following validation metrics. These are retained as exploratory historical evidence, not as a fair final comparison: the runs used different training histories/budgets, the LSTM result came from an earlier run, and seed coverage is incomplete.

| Model | Direction | R@1 | R@5 | R@10 | MRR | Median Rank |
|---|---|---:|---:|---:|---:|---:|
| LSTM | T2I | 3.4959% | 18.6394% | 28.4978% | 0.118431 | 33 |
| LSTM | I2T | 5.2890% | 23.7711% | 34.7031% | 0.148898 | 24 |
| GRU | T2I | 0.0118% | 0.1062% | 0.1848% | 0.001728 | 2534 |
| GRU | I2T | 0.0197% | 0.0983% | 0.2556% | 0.001660 | 3633 |
| Transformer | T2I | 0.0118% | 0.0786% | 0.2045% | 0.001696 | 2530 |
| Transformer | I2T | 0.0197% | 0.0983% | 0.1770% | 0.001643 | 3697 |
| Cross-Modal Attention | T2I | 0.0236% | 0.1770% | 0.3854% | 0.002866 | 1772 |
| Cross-Modal Attention | I2T | 0.0197% | 0.2359% | 0.4129% | 0.002613 | 3723 |

The historical results show why validation loss should not be treated as a substitute for retrieval metrics. They also indicate that retrieval performance is currently weak for several runs and that the final controlled comparison is necessary.

### Table 4 — Main experimental results (mean ± SD)

| Model | Direction | R@1 (mean ± SD) | R@5 (mean ± SD) | R@10 (mean ± SD) | Mean Recall (mean ± SD) | MRR (mean ± SD) |
|---|---|---|---|---|---|---|
| Baseline | T2I | Pending | Pending | Pending | Pending | Pending |
| Baseline | I2T | Pending | Pending | Pending | Pending | Pending |
| RNN | T2I / I2T | Pending | Pending | Pending | Pending | Pending |
| LSTM | T2I / I2T | Pending | Pending | Pending | Pending | Pending |
| GRU | T2I / I2T | Pending | Pending | Pending | Pending | Pending |
| Transformer | T2I / I2T | Pending | Pending | Pending | Pending | Pending |
| Cross-Modal Attention | T2I / I2T | Pending | Pending | Pending | Pending | Pending |

The table will be populated only after comparable runs for seeds 42, 123, and 456 have completed. T2I and I2T must be presented as separate rows in the final version.

## 14. Model Comparison

The models are compared on two dimensions:

1. **Retrieval quality:** Recall@1/5/10, Mean Recall, MRR, and Median Rank in both directions.
2. **Efficiency:** training time, inference latency, throughput, total/trainable parameters, and saved model size.

The current evidence is insufficient to declare a final winner. The preliminary seed-42 comparison favors LSTM by validation retrieval loss, but the historical retrieval metrics are uneven and the training histories are not fully aligned. The final model comparison will use the same fold, preprocessing, evaluation candidate set, training-step budget, and seed set wherever feasible.

For I2T, a query is counted as a successful retrieval at K if at least one of the five ground-truth captions for that image appears among the top K results. This definition must be retained when interpreting Recall@K. The evaluator's Mean Recall is the arithmetic mean of R@1, R@5, and R@10 for a direction.

## 15. Ablation Study

Ablation experiments are used to understand the contribution of individual components. An ablation claim is valid only when the compared configurations differ in the intended component and all other relevant settings are held fixed.

### 15.1 Cross-modal attention ablation (preliminary)

A GRU experiment compared attention disabled and attention enabled. The attention-enabled run also enabled the auxiliary cross-modal matching loss (weight 0.2), while the attention-disabled run did not. Therefore, the comparison is confounded and cannot isolate the effect of attention alone.

| Configuration | Training objective | Validation retrieval loss | Training time | Interpretation |
|---|---:|---:|---:|---|
| Attention disabled; matching loss disabled | 2.056058 | 1.815485 | 11.35 min | Reference configuration |
| Attention enabled; matching loss enabled (weight 0.2) | 2.199328 | 1.848958 | 26.51 min | Combined attention + auxiliary-loss change; not an isolated ablation |

### Table 5 — Ablation study

| Component removed/changed | Metric | Change vs. full model | Status / caveat |
|---|---|---|---|
| Cross-modal attention disabled | Validation retrieval loss and retrieval metrics | Preliminary loss comparison available; final retrieval-metric delta Pending | Confounded by matching-loss change |
| Auxiliary matching loss disabled | Matching metrics and retrieval metrics | Pending | Must be isolated from attention |
| Five captions → one caption | Retrieval metrics | Pending | Phase 5 study |
| No augmentation → light augmentation | Retrieval metrics | Pending | Phase 5 study |
| Pretrained ResNet50 → scratch initialization | Retrieval metrics | Pending | Phase 5 study |
| Frozen ResNet50 → last-block fine-tuning | Retrieval metrics and cost | Pending | Phase 5 study |

## 16. Error Analysis

Error analysis will use actual retrieval outputs from completed checkpoints. No illustrative or hypothetical example will be reported as a model failure.

For each selected query, the report will show the query image or caption, the top-five retrieved candidates, whether each candidate is relevant, and a short explanation of the observed error.

Candidate error categories include:

- **Object confusion:** retrieving images with a visually similar but incorrect object.
- **Action confusion:** confusing activities or interactions.
- **Attribute confusion:** incorrect color, size, material, or other visual attributes.
- **Spatial-relation confusion:** misunderstanding relative positions or interactions between objects.
- **Multiple-object omission:** matching one object while ignoring other important entities in the caption.
- **Semantic mismatch:** high-ranked candidates share superficial visual or lexical features but do not express the same meaning.

### Table 6 — Error analysis

| Error type | Count | Example (actual query and top results) | Likely cause |
|---|---:|---|---|
| Object confusion | Pending | Pending — extract from model predictions | Pending |
| Action confusion | Pending | Pending — extract from model predictions | Pending |
| Attribute confusion | Pending | Pending — extract from model predictions | Pending |
| Spatial-relation confusion | Pending | Pending — extract from model predictions | Pending |
| Multiple-object omission | Pending | Pending — extract from model predictions | Pending |
| Semantic mismatch | Pending | Pending — extract from model predictions | Pending |

The final analysis will include examples from both T2I and I2T. Counts will be computed from a documented sample or a defined evaluation subset rather than estimated.

## 17. Discussion

The current experiments show that training loss and retrieval quality must be interpreted separately. In the historical evaluation, the LSTM obtained substantially stronger retrieval metrics than several other architectures, while some models produced very low Recall@K and MRR. In the preliminary three-model comparison, LSTM also had the lowest validation retrieval loss, but the runs are not sufficient to establish a final ranking.

The cross-modal attention implementation introduces an additional matching branch. Since the current retrieval evaluator uses the independent modality embeddings, improvements in matching loss or matching accuracy do not automatically imply better retrieval ranking. The matching branch therefore requires separate evaluation, and any claim about retrieval improvement must be supported by a design in which the attention-derived representation participates in retrieval scoring.

The Phase 5 study investigates caption count, augmentation, and visual-backbone initialization/fine-tuning. Its results will help determine whether data strategy and backbone training influence the retrieval objective and ranking metrics. The short 200-step budget makes these experiments diagnostic rather than a substitute for final full-budget evaluation.

A reliable final conclusion requires repeated seeds, retrieval metrics in both directions, a held-out test evaluation after model selection, concrete error examples, and computational-cost measurements.

## 18. Limitations

1. Flickr30k is a relatively small dataset compared with large-scale image–text pretraining corpora.
2. Captions are subjective and may not mention every relevant object, attribute, or relation in an image.
3. Some current experiments use short step budgets and one seed; they are preliminary.
4. Historical model results were produced under different training histories and are not a controlled final comparison.
5. The cross-modal attention comparison is confounded when the auxiliary matching loss changes at the same time.
6. The current retrieval evaluator ranks independent image/text embeddings; it does not rank with the cross-modal matching score.
7. Complete inference latency, throughput, parameter counts, model sizes, and exact hardware specifications remain Pending.
8. Final held-out test evaluation and quantitative error analysis remain Pending.

## 19. Ethical Considerations

The project uses Flickr30k images obtained from Flickr. The project team does not own the image copyrights. Use of the images must comply with the Flickr Terms of Use and the dataset provider's stated research/educational non-commercial conditions [2]. Images must not be redistributed as though they were owned by the project team.

The system is a retrieval tool, not a source of verified facts about people or scenes. Retrieved captions can be incomplete, subjective, or incorrect. The interface should present ranked candidates as model predictions rather than guaranteed descriptions. Any public demonstration should avoid exposing personal or sensitive content unnecessarily and should preserve attribution and dataset-use restrictions.

## 20. Conclusion

VisionMatch AI implements a bidirectional image–text retrieval pipeline using a shared embedding space and contrastive training. The project compares a lightweight CNN/mean-pooling baseline with ResNet50-based models using recurrent and Transformer text encoders, and it includes an advanced cross-modal attention/matching branch.

The current evidence is preliminary. Historical retrieval evaluation indicates substantial differences among model configurations, while the short controlled seed-42 comparison shows the lowest validation retrieval loss for LSTM among the three tested configurations. These findings are not sufficient to select a final model. A fair multi-seed comparison, separate matching evaluation, final test evaluation, error analysis, and complete computational-cost measurements are still required.

## 21. Future Work

- Complete the controlled architecture comparison for seeds 42, 123, and 456.
- Evaluate all final candidates using T2I and I2T Recall@1/5/10, Mean Recall, MRR, and Median Rank.
- Select the final architecture using validation data and evaluate the held-out test set only after selection.
- Complete the Phase 5 caption-count, augmentation, pretrained-vs-scratch, and fine-tuning studies.
- Isolate cross-modal attention from the auxiliary matching-loss contribution.
- Evaluate the cross-modal matching branch using matching-specific metrics.
- Generate real top-five retrieval examples and categorize errors in both directions.
- Measure parameter counts, checkpoint/export sizes, inference latency, throughput, and memory use.
- Complete the web demo using a selected and evaluated checkpoint.
- Explore larger image–text pretrained encoders or improved negative-sampling strategies if time and compute allow.

## 22. References

[1] M. Hodosh, P. Young, and J. Hockenmaier, “Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics,” *Journal of Artificial Intelligence Research*, vol. 47, pp. 853–899, 2013. [JAIR publication](https://jair.org/index.php/jair/article/view/108ॉट).

[2] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, “From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions,” *Transactions of the Association for Computational Linguistics*, vol. 2, pp. 67–78, 2014. [Paper DOI](https://doi.org/10.1162/tacl_a_00166) · [Official Flickr30k dataset page](https://hockenmaier.cs.illinois.edu/DenotationGraph/data/).

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in *Proceedings of CVPR*, 2016, pp. 770–778. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

[4] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. [DOI](https://doi.org/10.1162/neco.1997.9.8.1735).

[5] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” in *Proceedings of EMNLP*, 2014, pp. 1724–1734. [ACL Anthology](https://aclanthology.org/D14-1179/).

[6] A. Vaswani et al., “Attention Is All You Need,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017. [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

[7] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” in *Proceedings of ICML*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020).

---



### Phase 5 — data strategy and visual-backbone studies (preliminary)

Phase 5 evaluates five GRU configurations on Flickr30k fold 1 (seed 42): (1) reference setup with five captions/image and frozen pretrained ResNet50; (2) one caption/image; (3) light image augmentation; (4) pretrained ResNet50 with its last block unfrozen; and (5) ResNet50 initialized from scratch with its last block trainable. Each run uses one epoch, 200 training steps, batch size 1, and 100 validation steps. The evaluation pipeline was updated to use batch size 4 while retaining the full 5,086-image validation split.

The reference configuration completed with training objective 0.819112, validation retrieval loss 0.805117, and training time 7.56 minutes. Its saved checkpoint was successfully reused during resume, avoiding repeated training; validation feature extraction was then run with the updated evaluation batch size. The one-caption configuration had started training in the latest captured log, but no completed metrics were available at that point. The remaining three configurations and final Phase 5 comparison are **Pending**. These short-budget validation losses are diagnostic and are not directly comparable with the separate three-model architecture comparison because the batch size, step budget, and evaluation configuration differ. Test data was not used.

## Appendix A — Reporting and reproducibility checklist

- [x] Define T2I and I2T retrieval tasks.
- [x] Record Flickr30k source, size, and usage restriction.
- [x] Document image and caption tensor preprocessing.
- [x] Document baseline and advanced architecture components from the implementation.
- [x] Record preliminary seed-42 training losses and times.
- [x] Record historical retrieval metrics with a non-comparability warning.
- [x] Document known confounds in the cross-modal attention ablation.
- [ ] Complete all comparable model runs for seeds 42, 123, and 456.
- [ ] Populate Table 4 with mean ± SD.
- [ ] Complete isolated ablations and populate Table 5.
- [ ] Generate real retrieval examples and populate Table 6.
- [ ] Measure model parameters, file size, inference latency, throughput, and memory.
- [ ] Populate Table 7 and document exact hardware/software environment.
- [ ] Run final held-out test evaluation after model selection.
- [ ] Add plots, retrieval galleries, and qualitative examples.
- [ ] Format and verify the final 15–25-page report.


### Progress update — repeated-seed comparison (20 November 2026)

The controlled three-model comparison is currently progressing through nine runs (GRU, LSTM, and cross-modal attention for seeds 42, 123, and 456). The following runs have completed:

| Seed | Model | Training objective | Validation retrieval loss | Training time |
|---:|---|---:|---:|---:|
| 42 | GRU | 2.086361 | 1.898083 | 25.16 min |
| 42 | LSTM | 2.089612 | 1.884016 | 22.03 min |
| 42 | Cross-modal attention (LSTM) | 2.235646 | 1.903820 | 55.62 min |
| 123 | GRU | 2.039485 | 1.826844 | 25.51 min |

The seed-123 LSTM run is currently in progress (run 5/9); no completed metrics are available for it yet. Seed-123 cross-modal attention and all seed-456 runs are also pending. The common setup is fold 1, batch size 4, one epoch, 200 training steps, 200 validation batches, AdamW (learning rate 0.0001), InfoNCE (temperature 0.07), five captions per image, no augmentation, and a frozen pretrained ResNet50. All listed runs executed on CPU, and the held-out test split was not used.

**Interpretation caution:** the cross-modal attention configuration additionally enables an auxiliary matching loss (weight 0.2), whereas GRU and LSTM do not. Therefore, this is not an isolated attention ablation. Also, the current Keras logs warn that projection layers receive and discard sequence-mask metadata. This warning has not stopped training, but its effect on padded-token handling should be investigated before finalizing the methodology and results.

These are validation-loss and training-time observations only. Retrieval-ranking metrics (T2I/I2T Recall@K, Mean Recall, MRR, and Median Rank) remain necessary for final model comparison; no final winning model is selected at this stage.


### Phase 5 progress update — one-caption experiment (20 November 2026)

The GRU one-caption-per-image experiment has completed on CPU. It used seed 42, fold 1, batch size 1, one epoch, 200 training steps, 100 validation batches, AdamW with learning rate 0.0001, InfoNCE temperature 0.07, a pretrained frozen ResNet50, and no image augmentation. Evaluation used batch size 4 over the validation set; the held-out test split was not used.

| Phase 5 configuration | Captions/image | Training objective | Validation retrieval loss | Training time |
|---|---:|---:|---:|---:|
| Reference | 5 | 0.819112 | 0.805117 | 7.56 min |
| One-caption | 1 | 0.000000 | 0.847462 | 7.66 min |

The one-caption run's reported training objective and retrieval loss are both exactly zero, while its validation retrieval loss is non-zero. This is an anomalous result and must not be interpreted as successful optimization. The loss computation and negative-sampling/batch construction for this configuration require code-level inspection before the experiment can support a conclusion about caption count. The validation-loss difference (0.847462 versus 0.805117) is recorded descriptively only; the one-caption result is flagged for verification.

The light-augmentation experiment (Phase 5 run 3/5) had started but had not produced completed metrics in the latest log. The pretrained fine-tuning and scratch-backbone experiments remain pending. These Phase 5 runs use a different budget and batch size from the main repeated-seed comparison and must remain a separate experiment group.



## Latest fair-comparison progress — 20 November 2026

This section records the newest run log received after the earlier Phase 5 updates. It is a progress update, not a final benchmark.

### LSTM — fold 1, seed 123 (completed)

The run `fair200_20261120_lstm_fold1_seed123` completed one epoch on CPU. Configuration: batch size 4, 200 training steps, 200 validation steps, AdamW, learning rate 0.0001, weight decay 0.0001, InfoNCE loss with temperature 0.07, pretrained frozen ResNet50, and five captions per image.

| Metric | Result |
|---|---:|
| Training objective / retrieval loss | 2.060249 |
| Validation retrieval loss | **1.856892** |
| Training time | 22.93 minutes |

Best checkpoint:
`checkpoints/experiments/fair200_20261120_lstm_fold1_seed123/best/best_model-1`

Latest checkpoint:
`checkpoints/experiments/fair200_20261120_lstm_fold1_seed123/latest/latest_checkpoint-1`

The test split was not used. The supplied log does not contain Recall@K, Mean Recall, MRR, or Median Rank for this run, so these retrieval metrics remain pending.

### Cross-Modal Attention — fold 1, seed 123 (in progress)

The next controlled-comparison run, `fair200_20261120_cross_attention_fold1_seed123` (run 6/9), started with one epoch, batch size 4, 200 training steps, and 200 validation steps. It uses an LSTM text encoder, Cross-Modal Attention, and an enabled image–caption matching branch with matching-loss weight 0.2. Optimizer: AdamW; learning rate: 0.0001; retrieval loss: InfoNCE; temperature: 0.07. The process is running on CPU.

At the time of this update, the run had entered its first epoch and was filling the TensorFlow shuffle buffer. No final training loss, validation loss, checkpoint, or retrieval metrics were available. All such results remain **Pending**.

### Interpretation and limitations

The Keras warnings state that projection layers do not preserve an attached sequence mask. They are warnings, not fatal errors: the LSTM run continued through training completion and saved both best and latest checkpoints. The warnings should nevertheless be reviewed as a potential modeling concern before finalizing the architecture.

The validation loss of 1.856892 is not a retrieval-ranking metric and cannot by itself establish that LSTM is the best retrieval model. The comparison remains incomplete until all planned runs finish and their validation retrieval metrics are evaluated consistently. The held-out test split remains untouched.


---

## 23. Explicit research hypotheses and how they will be tested

The project uses the following hypotheses as testable research claims. None should be described as accepted until the corresponding retrieval-ranking results support it.

**H1 — Sequence encoders versus the lightweight baseline.** Recurrent text encoders (RNN, GRU, or LSTM) are expected to improve bidirectional image–text retrieval over the CNN plus masked-mean text baseline by modeling word order and context. Test by comparing T2I and I2T Recall@1/5/10, Mean Recall, MRR, and Median Rank under a matched split and training budget.

**H2 — GRU versus LSTM.** GRU and LSTM are expected to show different quality/cost trade-offs. Compare the same visual backbone, text preprocessing, objective, batch size, step budget, and seeds; report retrieval metrics and runtime. A lower loss alone is not sufficient evidence of better ranking.

**H3 — Cross-modal attention.** Cross-modal attention is expected to improve retrieval by modeling image–text interactions. Test against a recurrent model with identical data, training schedule, and loss configuration. The existing comparison that enables matching loss only for the attention model is confounded and cannot isolate the effect of attention.

**H4 — Caption diversity and augmentation.** Multiple captions per image and suitable image augmentation may improve robustness and retrieval generalization. Test caption count and augmentation as separate factors, keeping the remaining configuration fixed. The one-caption pilot with a zero training objective must be debugged before drawing conclusions.

**H5 — Pretrained visual features.** Initializing ResNet50 from pretrained weights is expected to be more effective or computationally efficient than training from scratch on the available dataset. Compare under the same data split and a clearly stated fine-tuning policy.

## 24. Experimental controls and statistical reporting

For each final architecture claim, use the same fold, train/validation image split, preprocessing, retrieval candidate pool, evaluation implementation, optimizer budget, and tuning budget. Run at least three random seeds (42, 123, and 456) for every configuration presented as a final comparison. Publish the individual seed values and report mean ± standard deviation for every retrieval metric. If the run settings differ, label the experiments as separate diagnostic studies and do not combine them into one fair ranking.

Use validation data for checkpoint selection and hyperparameter decisions. Keep the held-out test split untouched until the final configuration is fixed; then evaluate it once and report the exact checkpoint and configuration used.

The current controlled-comparison plan covers GRU, LSTM, and Cross-Modal Attention for three seeds. It does not by itself provide a three-seed comparison for the baseline or Transformer. Either include those models under the same protocol or explicitly narrow the scope of the final architecture claim and explain the exclusion.

## 25. Ablation protocol

The following ablations are required to support causal claims:

1. **Attention ablation:** compare LSTM with and without cross-modal attention while keeping the matching-loss setting identical.
2. **Matching-loss ablation:** compare the same architecture with matching loss disabled/enabled while keeping attention and all other settings identical.
3. **Caption-count ablation:** compare one versus five captions per image; verify that every batch contains valid positives and negatives and that the loss is finite and non-zero when expected.
4. **Augmentation ablation:** compare no augmentation with the selected light augmentation under the same training budget.
5. **Backbone study:** compare pretrained frozen, pretrained fine-tuned, and scratch-initialized ResNet50, documenting which layers are trainable.

For each ablation, report validation retrieval metrics in both directions, training time, and the exact changed factor. Current results for several of these studies remain pending. The previously reported one-caption training objective of 0.000000 is flagged as anomalous and must not be interpreted until the loss/data pipeline is inspected.

## 26. Retrieval metrics, query groups, and error analysis

The primary metrics are Text-to-Image and Image-to-Text Recall@1, Recall@5, Recall@10, Mean Recall, MRR, and Median Rank. Report each seed separately and the mean ± standard deviation. Confidence intervals may additionally be estimated by a documented bootstrap over queries, but the resampling unit and number of replicates must be stated.

Conventional per-class accuracy is not inherently suitable because the task ranks image/caption candidates rather than assigning one class label. If trustworthy category annotations are available, report retrieval metrics by category. Otherwise, define query groups using a reproducible text/tag rule (for example, people, animals, actions, colors, and spatial relations) and report group-wise retrieval metrics with group sizes. Do not imply that these query groups are ground-truth classes.

The final error-analysis sample must contain real model outputs from both retrieval directions. For every sampled query, preserve the query, relevant ground-truth item(s), top-five retrieved candidates, ranks or similarity scores, and a manually reviewed error category: object confusion, action confusion, attribute mismatch, spatial-relation mismatch, omitted object, or semantic mismatch. Report the sample-selection method and category counts. The current error table is a template; its counts and examples remain pending.

## 27. Computational cost and deployment validation

For each final model, report total and trainable parameter counts, checkpoint size, training duration, inference latency and throughput for both retrieval directions, and peak memory when available. Record exact hardware and software: CPU model, GPU model and VRAM (if used), system RAM, operating system, Python, TensorFlow, CUDA/cuDNN versions. State the inference batch size, warm-up iterations, timed repetitions, and whether image/text embeddings are precomputed.

The FastAPI and web interface provide the demo pathway, but the final submission should include an end-to-end validation record using the selected checkpoint and its matching vocabulary/gallery. Verify text-to-image search, image-to-text search, invalid input handling, and that the displayed results correspond to the configured model.

## 28. Updated run status — 20 November 2026

The latest received log supersedes the earlier note that seed-123 LSTM was still running. The run `fair200_20261120_lstm_fold1_seed123` completed on CPU with one epoch, batch size 4, 200 training steps, and 200 validation steps. It used AdamW (learning rate 0.0001, weight decay 0.0001), InfoNCE (temperature 0.07), five captions per image, and a frozen pretrained ResNet50.

| Metric | Result |
|---|---:|
| Training objective / retrieval loss | 2.060249 |
| Validation retrieval loss | 1.856892 |
| Training time | 22.93 minutes |

Best checkpoint: `checkpoints/experiments/fair200_20261120_lstm_fold1_seed123/best/best_model-1`.

The seed-123 Cross-Modal Attention run had started but had not produced final metrics in the supplied log. The seed-456 runs were still pending in the latest confirmed progress. The supplied log did not report Recall@K, Mean Recall, MRR, or Median Rank for this LSTM run. Do not substitute validation loss for retrieval-ranking metrics.

## 29. Updated completion checklist

- [ ] Complete the three-seed runs for each architecture included in the final comparison.
- [ ] Include the baseline and Transformer in the controlled comparison, or explicitly limit and justify the scope.
- [ ] Calculate per-seed T2I/I2T Recall@K, Mean Recall, MRR, Median Rank, and mean ± standard deviation.
- [ ] Investigate Keras sequence-mask warnings and verify padded tokens cannot affect sequence representations.
- [ ] Debug and rerun the one-caption experiment with zero reported training objective.
- [ ] Complete attention-only and matching-loss-only ablations.
- [ ] Complete caption-count, augmentation, pretrained-vs-scratch, and fine-tuning studies.
- [ ] Populate the error-analysis table with actual queries and top-five predictions.
- [ ] Add query-group results or explain the absence of class-wise metrics.
- [ ] Record parameters, checkpoint size, inference latency/throughput, memory, and exact hardware.
- [ ] Select the model using validation only and run the held-out test evaluation once.
- [ ] Validate the live demo end to end with the final selected checkpoint and gallery.
- [ ] Add plots, galleries, and final reproducibility evidence.


## Latest Phase 5 GRU checkpoint evaluation progress — 20 November 2026

A validation evaluation was launched for `checkpoints/experiments/gru_phase5_resnet50_pretrained/best` with `--model-type full --encoder-type gru --pretrained`, writing to `results/evaluation_gru_resnet50.json`. The evaluator restored `best_model-1` successfully. The checkpoint metadata reports epoch 0 (zero-based), fold 1, seed 42, and validation loss **0.8051351488**.

The recorded training configuration is one epoch, batch size 1, AdamW (learning rate 0.0001, weight decay 0.0001), InfoNCE temperature 0.07, matching-loss weight 0.2, up to 200 training steps and 100 validation steps, with pretrained ResNet50. The supplied log showed validation feature extraction reaching 4,800 images, but it ended before the evaluator printed its final metrics. Consequently, this evaluation remains **in progress / output not yet verified**. No Recall@K, Mean Recall, MRR, Median Rank, final query count, or completed JSON artifact is claimed here.

The Keras warnings about projection layers discarding sequence-mask metadata should be investigated; they did not stop the checkpoint restore or the extraction progress shown. oneDNN and CPU feature notices are informational runtime messages. The checkpoint validation loss is not a substitute for retrieval-ranking metrics. Update this record only after the process exits and the JSON file is inspected. The held-out test set must remain unused until final model selection.

## Completed Phase 5 GRU validation evaluation — 20 November 2026

The evaluation for `checkpoints/experiments/gru_phase5_resnet50_pretrained/best` completed and saved `results/evaluation_gru_resnet50.json`. The evaluator restored `best_model-1` and extracted validation embeddings for 5,086 images. The log explicitly confirms: **Held-out test split used: NO**.

The checkpoint identifies fold 1, seed 42, epoch 0 (zero-based), stored validation loss 0.8051351488. The recorded configuration is a full model with GRU text encoder and pretrained ResNet50, one epoch, batch size 1, AdamW (learning rate 0.0001, weight decay 0.0001), InfoNCE temperature 0.07, matching-loss weight 0.2, maximum 200 training steps and 100 validation steps.

### Final validation retrieval metrics

| Direction | Recall@1 | Recall@5 | Recall@10 | Mean Recall | Median Rank | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Text-to-Image | 0.0197% | 0.1101% | 0.2320% | 0.1206% | 2,539 | 0.001861 |
| Image-to-Text | 0.0000% | 0.0983% | 0.1966% | 0.0983% | 3,921.5 | 0.001512 |

Text-to-Image: 25,430 caption queries, 5,086 image candidates. Image-to-Text: 5,086 image queries, 25,430 caption candidates.

**Analysis:** These retrieval scores are extremely low. In particular, Recall@10 remains below 0.25% in both directions, while median ranks are 2,539 and 3,921.5. The result aligns with the poor relevance observed in manual demo searches. The checkpoint restoration and evaluation pipeline work, but the current model does not yet provide useful retrieval quality. The stored validation loss (0.8051351488) must not be interpreted as evidence of good ranking performance.

Keras warned that `projection_network` and `text_projection` discard sequence-mask metadata. Evaluation completed despite these warnings, but they are a priority for investigation because padding-mask information may be lost downstream. oneDNN and CPU feature messages are informational runtime notices. The output JSON was saved at `results/evaluation_gru_resnet50.json`. The held-out test set remains untouched.

The earlier progress-only note for this run is superseded by these completed metrics. Do not add these numbers to a final test-results table; they are validation-only results from one short run and one seed.
