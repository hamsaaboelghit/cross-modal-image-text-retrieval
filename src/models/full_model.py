
import tensorflow as tf

try:
    from .image_encoder import ImageEncoder
    from .text_encoder import TextEncoder
    from .projection_head import CrossModalProjection
    from .cross_modal_attention import CrossModalAttention
except ImportError:
    from image_encoder import ImageEncoder
    from text_encoder import TextEncoder
    from projection_head import CrossModalProjection
    from cross_modal_attention import CrossModalAttention


class CrossModalRetrievalModel(tf.keras.Model):
    """
    Image-text retrieval model with an optional cross-modal
    matching branch.

    Image input:
        [B, H, W, 3], RGB in [0, 255].

    Caption input:
        [B, C, L], integer token IDs.

    Outputs:
        image_embeddings: [B, 512]
        text_embeddings: [B, C, 512]

    Optional matching outputs:
        matching_scores: [B, C]
        negative_matching_scores: [B, C]
        matching_negatives_valid: scalar bool
    """

    def __init__(
        self,
        vocab_size,
        encoder_type="lstm",
        embedding_dim=300,
        hidden_dim=512,
        num_layers=1,
        dropout=0.2,
        bidirectional=False,
        use_attention=True,
        pretrained=True,
        unfreeze_last_block=False,
        projection_dropout=0.1,
        use_cross_modal_attention=False,
        cross_modal_num_heads=8,
        cross_modal_dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_encoder = ImageEncoder(
            pretrained=pretrained,
            unfreeze_last_block=unfreeze_last_block,
            name="image_encoder",
        )

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            name="text_encoder",
        )

        self.projection = CrossModalProjection(
            image_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            embedding_dim=512,
            dropout=projection_dropout,
            name="cross_modal_projection",
        )

        self.embedding_dim = 512
        self.use_cross_modal_attention = bool(
            use_cross_modal_attention
        )

        if self.use_cross_modal_attention:
            self.cross_modal_attention = CrossModalAttention(
                text_dim=self.text_encoder.output_dim,
                image_dim=self.image_encoder.output_dim,
                embedding_dim=self.embedding_dim,
                num_heads=cross_modal_num_heads,
                dropout=cross_modal_dropout,
                name="cross_modal_attention",
            )

            self.matching_head = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.embedding_dim,
                        name="matching_hidden",
                    ),
                    tf.keras.layers.Activation(
                        "gelu",
                        name="matching_gelu",
                    ),
                    tf.keras.layers.Dropout(
                        cross_modal_dropout
                    ),
                    tf.keras.layers.Dense(
                        1,
                        name="matching_output",
                    ),
                ],
                name="matching_head",
            )
        else:
            self.cross_modal_attention = None
            self.matching_head = None

    def encode_image(self, images, training=None):
        images = tf.convert_to_tensor(images)

        tf.debugging.assert_rank(images, 4)
        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must have 3 RGB channels.",
        )
        tf.debugging.assert_positive(tf.shape(images)[0])

        image_features = self.image_encoder(
            images,
            training=training,
        )

        return self.projection.image_projection(
            image_features,
            training=training,
        )

    def encode_text(self, captions, training=None):
        captions = tf.convert_to_tensor(captions)

        tf.debugging.assert_rank(captions, 2)
        tf.debugging.assert_positive(tf.shape(captions)[0])
        tf.debugging.assert_positive(tf.shape(captions)[1])

        text_features = self.text_encoder(
            captions,
            training=training,
        )

        return self.projection.text_projection(
            text_features,
            training=training,
        )

    def _score_pairs(
        self,
        text_sequence_features,
        text_mask,
        image_regions,
        image_embeddings,
        training=None,
    ):
        """
        Score aligned image-caption pairs.

        All inputs have the same first dimension N:
            text_sequence_features: [N, L, text_dim]
            text_mask: [N, L]
            image_regions: [N, R, image_dim]
            image_embeddings: [N, 512]
        """

        fused_features, attention_weights = (
            self.cross_modal_attention(
                text_features=text_sequence_features,
                image_regions=image_regions,
                text_mask=text_mask,
                training=training,
            )
        )

        valid_mask = tf.cast(
            text_mask[:, :, tf.newaxis],
            fused_features.dtype,
        )

        summed_features = tf.reduce_sum(
            fused_features * valid_mask,
            axis=1,
        )

        valid_counts = tf.reduce_sum(
            valid_mask,
            axis=1,
        )

        pooled_features = summed_features / tf.maximum(
            valid_counts,
            tf.cast(1.0, fused_features.dtype),
        )

        matching_input = tf.concat(
            [
                image_embeddings,
                pooled_features,
            ],
            axis=-1,
        )

        matching_logits = self.matching_head(
            matching_input,
            training=training,
        )

        matching_logits = tf.squeeze(
            matching_logits,
            axis=-1,
        )

        return matching_logits, attention_weights

    def call(
        self,
        images,
        captions,
        training=None,
        return_attention=False,
    ):
        images = tf.convert_to_tensor(images)
        captions = tf.convert_to_tensor(captions)

        tf.debugging.assert_rank(
            images,
            4,
            message="Images must have shape [B,H,W,3].",
        )

        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must have 3 RGB channels.",
        )

        tf.debugging.assert_rank(
            captions,
            3,
            message="Captions must have shape [B,C,L].",
        )

        batch_size = tf.shape(images)[0]
        caption_batch_size = tf.shape(captions)[0]
        num_captions = tf.shape(captions)[1]
        sequence_length = tf.shape(captions)[2]

        tf.debugging.assert_equal(
            batch_size,
            caption_batch_size,
            message="Image and caption batch sizes must match.",
        )

        tf.debugging.assert_positive(batch_size)
        tf.debugging.assert_positive(num_captions)
        tf.debugging.assert_positive(sequence_length)

        # --------------------------------------------------
        # 1. Image encoding
        # --------------------------------------------------
        if self.use_cross_modal_attention:
            image_features, image_regions = self.image_encoder(
                images,
                training=training,
                return_spatial=True,
            )
        else:
            image_features = self.image_encoder(
                images,
                training=training,
            )
            image_regions = None

        # --------------------------------------------------
        # 2. Flatten captions: [B,C,L] -> [B*C,L]
        # --------------------------------------------------
        flat_captions = tf.reshape(
            captions,
            (
                batch_size * num_captions,
                sequence_length,
            ),
        )

        # --------------------------------------------------
        # 3. Text encoding
        # --------------------------------------------------
        if self.use_cross_modal_attention:
            if return_attention:
                (
                    text_features,
                    text_sequence_features,
                    attention_weights,
                ) = self.text_encoder(
                    flat_captions,
                    training=training,
                    return_attention=True,
                    return_sequence=True,
                )
            else:
                (
                    text_features,
                    text_sequence_features,
                ) = self.text_encoder(
                    flat_captions,
                    training=training,
                    return_sequence=True,
                )
                attention_weights = None
        else:
            if return_attention:
                text_features, attention_weights = (
                    self.text_encoder(
                        flat_captions,
                        training=training,
                        return_attention=True,
                    )
                )
            else:
                text_features = self.text_encoder(
                    flat_captions,
                    training=training,
                )
                attention_weights = None

            text_sequence_features = None

        # --------------------------------------------------
        # 4. Independent retrieval embeddings
        # --------------------------------------------------
        image_embeddings, flat_text_embeddings = self.projection(
            image_features,
            text_features,
            training=training,
        )

        text_embeddings = tf.reshape(
            flat_text_embeddings,
            (
                batch_size,
                num_captions,
                self.embedding_dim,
            ),
        )

        output = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }

        # --------------------------------------------------
        # 5. Cross-modal matching branch
        # --------------------------------------------------
        if self.use_cross_modal_attention:
            text_mask = tf.not_equal(
                flat_captions,
                0,
            )

            paired_image_regions = tf.repeat(
                image_regions,
                repeats=num_captions,
                axis=0,
            )

            paired_image_embeddings = tf.repeat(
                image_embeddings,
                repeats=num_captions,
                axis=0,
            )

            positive_scores, cross_attention_weights = (
                self._score_pairs(
                    text_sequence_features=text_sequence_features,
                    text_mask=text_mask,
                    image_regions=paired_image_regions,
                    image_embeddings=paired_image_embeddings,
                    training=training,
                )
            )

            # Negative pairs:
            # Pair each caption with another image in the batch.
            # A one-item batch is marked invalid by the loss.
            negative_image_regions = tf.roll(
                image_regions,
                shift=1,
                axis=0,
            )

            negative_image_embeddings = tf.roll(
                image_embeddings,
                shift=1,
                axis=0,
            )

            paired_negative_regions = tf.repeat(
                negative_image_regions,
                repeats=num_captions,
                axis=0,
            )

            paired_negative_embeddings = tf.repeat(
                negative_image_embeddings,
                repeats=num_captions,
                axis=0,
            )

            negative_scores, _ = self._score_pairs(
                text_sequence_features=text_sequence_features,
                text_mask=text_mask,
                image_regions=paired_negative_regions,
                image_embeddings=paired_negative_embeddings,
                training=training,
            )

            output["matching_scores"] = tf.reshape(
                positive_scores,
                (batch_size, num_captions),
            )

            output["negative_matching_scores"] = tf.reshape(
                negative_scores,
                (batch_size, num_captions),
            )

            output["matching_negatives_valid"] = (
                tf.greater(batch_size, 1)
            )

            if return_attention:
                output["cross_modal_attention_weights"] = (
                    tf.reshape(
                        cross_attention_weights,
                        (
                            batch_size,
                            num_captions,
                            sequence_length,
                            tf.shape(image_regions)[1],
                        ),
                    )
                )

        # --------------------------------------------------
        # 6. Text attention output
        # --------------------------------------------------
        if return_attention:
            if attention_weights is not None:
                output["attention_weights"] = tf.reshape(
                    attention_weights,
                    (
                        batch_size,
                        num_captions,
                        sequence_length,
                    ),
                )
            else:
                output["attention_weights"] = None

        return output


def _run_full_model_tests():
    tf.random.set_seed(42)

    print("Full Model Tests")
    print("-" * 60)
    print(f"TensorFlow: {tf.__version__}")

    batch_size = 2
    num_captions = 3
    sequence_length = 12
    vocab_size = 1000

    images = tf.random.uniform(
        (batch_size, 224, 224, 3),
        minval=0,
        maxval=256,
        dtype=tf.float32,
    )

    captions = tf.random.uniform(
        (batch_size, num_captions, sequence_length),
        minval=4,
        maxval=vocab_size,
        dtype=tf.int32,
    )

    captions_array = captions.numpy()
    captions_array[0, 1, 8:] = 0
    captions_array[1, 2, 5:] = 0
    captions = tf.convert_to_tensor(captions_array)

    model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="gru",
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        pretrained=False,
        use_cross_modal_attention=False,
    )

    outputs = model(
        images,
        captions,
        training=False,
        return_attention=True,
    )

    assert tuple(outputs["image_embeddings"].shape) == (
        batch_size,
        512,
    )

    assert tuple(outputs["text_embeddings"].shape) == (
        batch_size,
        num_captions,
        512,
    )

    tf.debugging.assert_near(
        tf.norm(outputs["image_embeddings"], axis=-1),
        tf.ones((batch_size,)),
        atol=1e-5,
    )

    tf.debugging.assert_near(
        tf.norm(outputs["text_embeddings"], axis=-1),
        tf.ones((batch_size, num_captions)),
        atol=1e-5,
    )

    assert tuple(outputs["attention_weights"].shape) == (
        batch_size,
        num_captions,
        sequence_length,
    )

    print("Independent retrieval embeddings: PASS")
    print("Embedding normalization: PASS")
    print("Text attention output: PASS")

    image_only = model.encode_image(
        images,
        training=False,
    )

    text_only = model.encode_text(
        tf.reshape(captions, (-1, sequence_length)),
        training=False,
    )

    assert tuple(image_only.shape) == (batch_size, 512)
    assert tuple(text_only.shape) == (
        batch_size * num_captions,
        512,
    )

    print("Independent image encoding: PASS")
    print("Independent text encoding: PASS")

    advanced_model = CrossModalRetrievalModel(
        vocab_size=vocab_size,
        encoder_type="gru",
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
        use_attention=True,
        pretrained=False,
        use_cross_modal_attention=True,
    )

    with tf.GradientTape() as tape:
        advanced_outputs = advanced_model(
            images,
            captions,
            training=True,
            return_attention=True,
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )

        positive_loss = loss_fn(
            tf.ones_like(
                advanced_outputs["matching_scores"]
            ),
            advanced_outputs["matching_scores"],
        )

        negative_loss = loss_fn(
            tf.zeros_like(
                advanced_outputs["negative_matching_scores"]
            ),
            advanced_outputs["negative_matching_scores"],
        )

        total_loss = positive_loss + negative_loss

    gradients = tape.gradient(
        total_loss,
        advanced_model.trainable_variables,
    )

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "Non-finite cross-modal gradient.",
        )

    assert tuple(
        advanced_outputs["matching_scores"].shape
    ) == (batch_size, num_captions)

    assert tuple(
        advanced_outputs["negative_matching_scores"].shape
    ) == (batch_size, num_captions)

    assert bool(
        advanced_outputs["matching_negatives_valid"].numpy()
    )

    assert tuple(
        advanced_outputs["cross_modal_attention_weights"].shape
    ) == (
        batch_size,
        num_captions,
        sequence_length,
        49,
    )

    print("Positive matching scores: PASS")
    print("Negative matching scores: PASS")
    print("Cross-modal attention output: PASS")
    print("Cross-modal gradient flow: PASS")

    # Single-image batches must be marked as having no valid negatives.
    single_outputs = advanced_model(
        images[:1],
        captions[:1],
        training=False,
    )

    assert not bool(
        single_outputs["matching_negatives_valid"].numpy()
    )

    print("Single-image negative-pair handling: PASS")

    print("-" * 60)
    print("All Full Model tests passed!")


if __name__ == "__main__":
    _run_full_model_tests()
