
import torch
import torch.nn as nn

try:
    from .image_encoder import ImageEncoder
    from .text_encoder import TextEncoder
    from .projection_head import CrossModalProjection
except ImportError:
    from image_encoder import ImageEncoder
    from text_encoder import TextEncoder
    from projection_head import CrossModalProjection


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

        # 3. Projection Heads
        self.projection = CrossModalProjection(
            image_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            embedding_dim=512,
            dropout=projection_dropout,
        )

        self.embedding_dim = 512

    def forward(self, images, captions, return_attention=False):

        if images.ndim != 4:
            raise ValueError(
                "Images must have shape [B, 3, H, W]."
            )

        if captions.ndim != 3:
            raise ValueError(
                "Captions must have shape [B, C, L]."
            )

        batch_size = images.size(0)
        num_captions = captions.size(1)

        if captions.size(0) != batch_size:
            raise ValueError(
                "Images and captions must have the same batch size."
            )

        # Image branch
        image_features = self.image_encoder(images)

        # Text branch
        # [B, C, L] -> [B*C, L]
        flat_captions = captions.reshape(
            batch_size * num_captions,
            captions.size(2),
        )

        if return_attention:
            text_features, attention_weights = self.text_encoder(
                flat_captions,
                return_attention=True,
            )
        else:
            text_features = self.text_encoder(flat_captions)
            attention_weights = None

        # Projection
        image_embeddings, flat_text_embeddings = self.projection(
            image_features,
            text_features,
        )

        # [B*C, 512] -> [B, C, 512]
        text_embeddings = flat_text_embeddings.reshape(
            batch_size,
            num_captions,
            self.embedding_dim,
        )

        output = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }

        # Return attention weights when available
        if return_attention:
            if attention_weights is not None:
                output["attention_weights"] = attention_weights.reshape(
                    batch_size,
                    num_captions,
                    captions.size(2),
                )
            else:
                output["attention_weights"] = None

        return output
