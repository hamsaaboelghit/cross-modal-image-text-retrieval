
import torch
import torch.nn as nn

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


class CrossModalRetrievalModel(nn.Module):

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
    ):
        super().__init__()

        # 1. Image Encoder
        self.image_encoder = ImageEncoder(
            pretrained=pretrained,
            unfreeze_last_block=unfreeze_last_block,
        )

        # 2. Text Encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            encoder_type=encoder_type,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
        )

        # 3. Independent Projection Heads
        self.projection = CrossModalProjection(
            image_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            embedding_dim=512,
            dropout=projection_dropout,
        )

        self.embedding_dim = 512

        # 4. Optional Advanced Cross-Modal Matching Branch
        self.use_cross_modal_attention = use_cross_modal_attention

        if self.use_cross_modal_attention:

            self.cross_modal_attention = CrossModalAttention(
                text_dim=self.text_encoder.output_dim,
                image_dim=self.image_encoder.output_dim,
                embedding_dim=self.embedding_dim,
                num_heads=cross_modal_num_heads,
                dropout=cross_modal_dropout,
            )

            # Pairwise image-caption matching head
            self.matching_head = nn.Sequential(
                nn.Linear(
                    self.embedding_dim * 2,
                    self.embedding_dim,
                ),
                nn.GELU(),
                nn.Dropout(cross_modal_dropout),
                nn.Linear(self.embedding_dim, 1),
            )

        else:
            self.cross_modal_attention = None
            self.matching_head = None

    def forward(
        self,
        images,
        captions,
        return_attention=False,
    ):

        # Input Validation

        if images.ndim != 4:
            raise ValueError(
                "Images must have shape [B, 3, H, W]."
            )

        if images.size(1) != 3:
            raise ValueError(
                "Images must have exactly 3 RGB channels."
            )

        if captions.ndim != 3:
            raise ValueError(
                "Captions must have shape [B, C, L]."
            )

        batch_size = images.size(0)
        num_captions = captions.size(1)
        sequence_length = captions.size(2)

        if captions.size(0) != batch_size:
            raise ValueError(
                "Images and captions must have the same batch size."
            )

        if num_captions == 0:
            raise ValueError(
                "At least one caption per image is required."
            )

        if sequence_length == 0:
            raise ValueError(
                "Caption sequence length must be greater than zero."
            )

        # 1. Image Encoding
        if self.use_cross_modal_attention:

            image_features, image_regions = self.image_encoder(
                images,
                return_spatial=True,
            )

        else:

            image_features = self.image_encoder(images)
            image_regions = None

        # 2. Flatten Captions
        flat_captions = captions.reshape(
            batch_size * num_captions,
            sequence_length,
        )

        # 3. Text Encoding
        if self.use_cross_modal_attention:

            if return_attention:

                (
                    text_features,
                    text_sequence_features,
                    attention_weights,
                ) = self.text_encoder(
                    flat_captions,
                    return_attention=True,
                    return_sequence=True,
                )

            else:

                text_features, text_sequence_features = (
                    self.text_encoder(
                        flat_captions,
                        return_sequence=True,
                    )
                )

                attention_weights = None

        else:

            if return_attention:

                text_features, attention_weights = (
                    self.text_encoder(
                        flat_captions,
                        return_attention=True,
                    )
                )

            else:

                text_features = self.text_encoder(flat_captions)
                attention_weights = None

            text_sequence_features = None

        # 4. Independent Retrieval Embeddings
        image_embeddings, flat_text_embeddings = self.projection(
            image_features,
            text_features,
        )

        text_embeddings = flat_text_embeddings.reshape(
            batch_size,
            num_captions,
            self.embedding_dim,
        )

        output = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }

        # 5. Optional Cross-Modal Matching
        if self.use_cross_modal_attention:

            # Each image is paired with its corresponding captions.
            paired_image_regions = image_regions.repeat_interleave(
                num_captions,
                dim=0,
            )

            paired_image_embeddings = image_embeddings.repeat_interleave(
                num_captions,
                dim=0,
            )

            # True = valid token, False = padding.
            text_mask = flat_captions != 0

            # Text-to-image cross-attention.
            fused_features, cross_attention_weights = (
                self.cross_modal_attention(
                    text_features=text_sequence_features,
                    image_regions=paired_image_regions,
                    text_mask=text_mask,
                )
            )

            # Masked mean pooling over valid text tokens.
            valid_mask = text_mask.unsqueeze(-1).to(
                fused_features.dtype
            )

            fused_pooled = (
                (fused_features * valid_mask).sum(dim=1)
                / valid_mask.sum(dim=1).clamp(min=1)
            )

            # Pairwise matching score.
            matching_input = torch.cat(
                [
                    paired_image_embeddings,
                    fused_pooled,
                ],
                dim=1,
            )

            matching_scores = self.matching_head(
                matching_input
            ).squeeze(-1)

            output["matching_scores"] = matching_scores.reshape(
                batch_size,
                num_captions,
            )

            if return_attention:

                output["cross_modal_attention_weights"] = (
                    cross_attention_weights.reshape(
                        batch_size,
                        num_captions,
                        sequence_length,
                        image_regions.size(1),
                    )
                )

        # 6. Original Text Attention Output
        if return_attention:

            if attention_weights is not None:

                output["attention_weights"] = (
                    attention_weights.reshape(
                        batch_size,
                        num_captions,
                        sequence_length,
                    )
                )

            else:

                output["attention_weights"] = None

        return output
