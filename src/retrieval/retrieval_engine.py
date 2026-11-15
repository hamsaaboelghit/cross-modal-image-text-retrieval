
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class RetrievalEngine:
    """
    Cross-modal image-text retrieval engine.

    Supports:
        1. Text-to-Image retrieval.
        2. Image-to-Text retrieval.
        3. Similarity matrix computation.

    Image embeddings:
        [num_images, embedding_dim]

    Text embeddings:
        [num_images, captions_per_image, embedding_dim]
    """

    def __init__(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_names: Optional[List[str]] = None,
        captions: Optional[List[List[str]]] = None,
    ):

        if not isinstance(image_embeddings, torch.Tensor):
            raise TypeError("image_embeddings must be a torch.Tensor.")

        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError("text_embeddings must be a torch.Tensor.")

        if image_embeddings.ndim != 2:
            raise ValueError(
                "image_embeddings must have shape [N, D]."
            )

        if text_embeddings.ndim != 3:
            raise ValueError(
                "text_embeddings must have shape [N, C, D]."
            )

        if image_embeddings.size(0) == 0:
            raise ValueError("At least one image is required.")

        if text_embeddings.size(1) == 0:
            raise ValueError("At least one caption per image is required.")

        if image_embeddings.size(0) != text_embeddings.size(0):
            raise ValueError(
                "Image and text embeddings must have the same "
                "number of images."
            )

        if image_embeddings.size(1) != text_embeddings.size(2):
            raise ValueError(
                "Image and text embedding dimensions must match."
            )

        if not torch.isfinite(image_embeddings).all():
            raise ValueError("Image embeddings contain invalid values.")

        if not torch.isfinite(text_embeddings).all():
            raise ValueError("Text embeddings contain invalid values.")

        self.device = image_embeddings.device

        self.image_embeddings = F.normalize(
            image_embeddings.detach().float(),
            p=2,
            dim=-1,
        )

        self.text_embeddings = F.normalize(
            text_embeddings.detach().float().to(self.device),
            p=2,
            dim=-1,
        )

        self.num_images = image_embeddings.size(0)
        self.captions_per_image = text_embeddings.size(1)
        self.embedding_dim = image_embeddings.size(1)

        if image_names is None:
            self.image_names = [
                f"image_{index}"
                for index in range(self.num_images)
            ]
        else:
            if len(image_names) != self.num_images:
                raise ValueError(
                    "image_names length must match number of images."
                )

            self.image_names = list(image_names)

        if captions is not None:

            if len(captions) != self.num_images:
                raise ValueError(
                    "captions must contain one list per image."
                )

            if any(
                len(image_captions) != self.captions_per_image
                for image_captions in captions
            ):
                raise ValueError(
                    "Every image must have the same number of captions."
                )

        self.captions = captions

        # Flatten text embeddings while preserving image-caption mapping.
        self.flat_text_embeddings = self.text_embeddings.reshape(
            self.num_images * self.captions_per_image,
            self.embedding_dim,
        )

    def compute_similarity_matrix(self):
        """
        Compute similarity between every image and every caption.

        Returns:
            Tensor of shape [num_images, num_images * captions_per_image].
        """

        return (
            self.image_embeddings
            @ self.flat_text_embeddings.T
        )

    def text_to_image(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict]:

        """
        Retrieve images for one or more text queries.

        query_embedding:
            [D] or [Q, D]

        Returns:
            One ranked result list per query.
        """

        if not isinstance(query_embedding, torch.Tensor):
            raise TypeError("query_embedding must be a torch.Tensor.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)

        if query_embedding.ndim != 2:
            raise ValueError(
                "Text query must have shape [D] or [Q, D]."
            )

        return self._retrieve(
            query_embedding,
            self.image_embeddings,
            top_k,
            self.image_names,
            "image",
        )

    def image_to_text(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict]:

        """
        Retrieve captions for one or more image queries.

        query_embedding:
            [D] or [Q, D]

        Returns:
            One ranked result list per query.
        """

        if not isinstance(query_embedding, torch.Tensor):
            raise TypeError("query_embedding must be a torch.Tensor.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)

        if query_embedding.ndim != 2:
            raise ValueError(
                "Image query must have shape [D] or [Q, D]."
            )

        caption_labels = []

        for image_index in range(self.num_images):

            for caption_index in range(self.captions_per_image):

                caption_text = None

                if self.captions is not None:
                    caption_text = self.captions[
                        image_index
                    ][caption_index]

                caption_labels.append({
                    "image_name": self.image_names[image_index],
                    "image_index": image_index,
                    "caption_index": caption_index,
                    "caption": caption_text,
                })

        return self._retrieve(
            query_embedding,
            self.flat_text_embeddings,
            top_k,
            caption_labels,
            "caption",
        )

    def _retrieve(
        self,
        query_embeddings,
        candidate_embeddings,
        top_k,
        candidate_labels,
        result_type,
    ):

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if query_embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                "Query embedding dimension does not match index."
            )

        if not torch.isfinite(query_embeddings).all():
            raise ValueError(
                "Query embeddings contain invalid values."
            )

        query_embeddings = F.normalize(
            query_embeddings.detach().float().to(self.device),
            p=2,
            dim=-1,
        )

        candidate_embeddings = candidate_embeddings.to(self.device)

        similarity = query_embeddings @ candidate_embeddings.T

        effective_k = min(top_k, candidate_embeddings.size(0))

        scores, indices = torch.topk(
            similarity,
            k=effective_k,
            dim=1,
        )

        all_results = []

        for query_index in range(query_embeddings.size(0)):

            query_results = []

            for score, candidate_index in zip(
                scores[query_index],
                indices[query_index],
            ):

                index = candidate_index.item()
                label = candidate_labels[index]

                query_results.append({
                    "type": result_type,
                    "score": score.item(),
                    "index": index,
                    "result": label,
                })

            all_results.append(query_results)

        return all_results
