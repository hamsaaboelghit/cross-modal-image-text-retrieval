# Phase 5 – Data Strategy and Pretrained vs Scratch

## Implemented controls

### Caption-count ablation
- Training supports one caption or all five captions per image.
- The one-caption selection is deterministic for a given seed and fixed per image throughout a run.
- Validation/test require all five captions. Caption ablations cannot silently change evaluation references.
- The default remains five captions.

### Image augmentation
- `none`: existing RGB resize pipeline.
- `light`: stateless random brightness, contrast, and saturation on training batches only.
- Values are clipped to [0, 255] to preserve the ResNet50 preprocessing contract.
- No horizontal flip is used because flipping may alter image-caption semantics.
- Validation/test are never augmented.

### Negative sampling
Triplet loss supports:
- `random`: choose a caption from a different image in the batch.
- `hardest`: choose the most similar caption embedding whose owner is a different image.
The selected strategy is recorded in TrainingConfig and passed through the shared runner. This ablation is specific to Triplet loss; InfoNCE and Margin keep their existing objectives.

### ResNet50 initialization
The full-model runner supports ImageNet initialization or random initialization. For a controlled comparison, `train_all_backbone=True` enables convolutional backbone training for both conditions while keeping BatchNormalization layers frozen. The pretrained and scratch runs must otherwise share architecture, data, seed, optimizer, learning rate, batch size, epochs, loss, caption count, and augmentation.

## Experiment matrix

| Experiment | Captions | Augmentation | Objective | ResNet50 |
|---|---:|---|---|---|
| data_reference | 5 | none | InfoNCE | pretrained, frozen |
| one_caption | 1 | none | InfoNCE | pretrained, frozen |
| light_augmentation | 5 | light | InfoNCE | pretrained, frozen |
| triplet_random | 5 | none | Triplet / random | pretrained, frozen |
| triplet_hardest | 5 | none | Triplet / hardest | pretrained, frozen |
| resnet50_pretrained_finetune | 5 | none | InfoNCE | pretrained, trainable |
| resnet50_scratch_finetune | 5 | none | InfoNCE | random, trainable |

Initial screening uses fold 1 and seed 42. Compare validation retrieval loss and bidirectional Recall@1/5/10, MRR, and Mean Recall. Do not use the test split for selection. Phase 6 should repeat selected comparisons with multiple seeds.

## Interpretation notes
- One-vs-five captions measures training-description quantity; validation always retains five references.
- Hard-negative mining may improve informative learning but can create noisier gradients.
- Scratch ResNet50 may need more epochs to converge. A same-epoch comparison is a controlled screening experiment, not proof that scratch has converged.
- Code preparation is not a completed experiment. No result is claimed until logs, checkpoints, and validation metrics exist.
