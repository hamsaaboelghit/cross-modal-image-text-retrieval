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
