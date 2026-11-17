
from typing import Dict, List, Optional

import tensorflow as tf


class RetrievalEngine:
    """
    TensorFlow cross-modal retrieval engine.

    Supports:
        1. Text-to-Image retrieval.
        2. Image-to-Text retrieval.
        3. Similarity matrix computation.

    Image embeddings: [N, D]
    Text embeddings: [N, C, D]
    """

    def __init__(
        self,
        image_embeddings,
        text_embeddings,
        image_names: Optional[List[str]] = None,
        captions: Optional[List[List[str]]] = None,
    ):
        self.image_embeddings = tf.convert_to_tensor(
            image_embeddings,
            dtype=tf.float32,
        )
        self.text_embeddings = tf.convert_to_tensor(
            text_embeddings,
            dtype=tf.float32,
        )

        tf.debugging.assert_rank(self.image_embeddings, 2)
        tf.debugging.assert_rank(self.text_embeddings, 3)

        tf.debugging.assert_positive(
            tf.shape(self.image_embeddings)[0],
            message="At least one image is required.",
        )
        tf.debugging.assert_positive(
            tf.shape(self.text_embeddings)[1],
            message="At least one caption per image is required.",
        )

        tf.debugging.assert_equal(
            tf.shape(self.image_embeddings)[0],
            tf.shape(self.text_embeddings)[0],
            message="Image and text counts must match.",
        )
        tf.debugging.assert_equal(
            tf.shape(self.image_embeddings)[1],
            tf.shape(self.text_embeddings)[2],
            message="Embedding dimensions must match.",
        )

        tf.debugging.assert_all_finite(
            self.image_embeddings,
            "Image embeddings contain invalid values.",
        )
        tf.debugging.assert_all_finite(
            self.text_embeddings,
            "Text embeddings contain invalid values.",
        )

        self.image_embeddings = tf.math.l2_normalize(
            self.image_embeddings,
            axis=-1,
        )
        self.text_embeddings = tf.math.l2_normalize(
            self.text_embeddings,
            axis=-1,
        )

        self.num_images = int(
            tf.shape(self.image_embeddings)[0].numpy()
        )
        self.captions_per_image = int(
            tf.shape(self.text_embeddings)[1].numpy()
        )
        self.embedding_dim = int(
            tf.shape(self.image_embeddings)[1].numpy()
        )

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

            self.image_names = [
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in image_names
            ]

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

        self.flat_text_embeddings = tf.reshape(
            self.text_embeddings,
            (
                self.num_images * self.captions_per_image,
                self.embedding_dim,
            ),
        )

    def compute_similarity_matrix(self):
        """
        Compute similarity between every image and every caption.

        Returns:
            Tensor [N, N*C].
        """
        return tf.matmul(
            self.image_embeddings,
            self.flat_text_embeddings,
            transpose_b=True,
        )

    def text_to_image(
        self,
        query_embedding,
        top_k: int = 5,
    ) -> List[List[Dict]]:
        """Retrieve images using one or more text embeddings."""

        query_embedding = tf.convert_to_tensor(
            query_embedding,
            dtype=tf.float32,
        )

        if query_embedding.shape.rank == 1:
            query_embedding = tf.expand_dims(query_embedding, axis=0)

        if query_embedding.shape.rank != 2:
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
        query_embedding,
        top_k: int = 5,
    ) -> List[List[Dict]]:
        """Retrieve captions using one or more image embeddings."""

        query_embedding = tf.convert_to_tensor(
            query_embedding,
            dtype=tf.float32,
        )

        if query_embedding.shape.rank == 1:
            query_embedding = tf.expand_dims(query_embedding, axis=0)

        if query_embedding.shape.rank != 2:
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

        tf.debugging.assert_equal(
            tf.shape(query_embeddings)[1],
            self.embedding_dim,
            message="Query embedding dimension does not match index.",
        )

        tf.debugging.assert_all_finite(
            query_embeddings,
            "Query embeddings contain invalid values.",
        )

        query_embeddings = tf.math.l2_normalize(
            query_embeddings,
            axis=-1,
        )

        similarity = tf.matmul(
            query_embeddings,
            candidate_embeddings,
            transpose_b=True,
        )

        effective_k = min(
            top_k,
            int(tf.shape(candidate_embeddings)[0].numpy()),
        )

        scores, indices = tf.math.top_k(
            similarity,
            k=effective_k,
            sorted=True,
        )

        scores = scores.numpy()
        indices = indices.numpy()

        all_results = []

        for query_index in range(len(indices)):
            query_results = []

            for score, candidate_index in zip(
                scores[query_index],
                indices[query_index],
            ):
                index = int(candidate_index)

                query_results.append({
                    "type": result_type,
                    "score": float(score),
                    "index": index,
                    "result": candidate_labels[index],
                })

            all_results.append(query_results)

        return all_results


def _run_retrieval_engine_tests():
    print("TensorFlow Retrieval Engine Tests")
    print("-" * 50)

    tf.random.set_seed(42)

    num_images = 10
    captions_per_image = 5
    embedding_dim = 32

    image_embeddings = tf.math.l2_normalize(
        tf.random.normal((num_images, embedding_dim)),
        axis=-1,
    )

    text_embeddings = tf.math.l2_normalize(
        tf.random.normal(
            (num_images, captions_per_image, embedding_dim)
        ),
        axis=-1,
    )

    image_names = [
        f"image_{index}.jpg"
        for index in range(num_images)
    ]

    captions = [
        [
            f"Caption {caption_index} for image {image_index}"
            for caption_index in range(captions_per_image)
        ]
        for image_index in range(num_images)
    ]

    engine = RetrievalEngine(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_names=image_names,
        captions=captions,
    )

    similarity = engine.compute_similarity_matrix()

    assert tuple(similarity.shape) == (
        num_images,
        num_images * captions_per_image,
    )

    print("Similarity matrix: PASS")

    text_results = engine.text_to_image(
        text_embeddings[0, 0],
        top_k=3,
    )

    assert len(text_results) == 1
    assert len(text_results[0]) == 3
    assert text_results[0][0]["type"] == "image"

    print("Text-to-Image retrieval: PASS")

    image_results = engine.image_to_text(
        image_embeddings[0],
        top_k=5,
    )

    assert len(image_results) == 1
    assert len(image_results[0]) == 5
    assert image_results[0][0]["type"] == "caption"

    print("Image-to-Text retrieval: PASS")

    batch_results = engine.text_to_image(
        text_embeddings[:2, 0],
        top_k=2,
    )

    assert len(batch_results) == 2

    print("Batch retrieval: PASS")
    print("-" * 50)
    print("All Retrieval Engine tests passed!")


if __name__ == "__main__":
    _run_retrieval_engine_tests()
