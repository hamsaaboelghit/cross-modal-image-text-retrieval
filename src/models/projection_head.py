
import tensorflow as tf


class ProjectionHead(tf.keras.layers.Layer):
    """
    Projects input features into a shared embedding space.

    Supports inputs with shape:
        [batch, feature_dim]
        [batch, sequence_length, feature_dim]

    The final embeddings are L2-normalized along the last dimension.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim=512,
        hidden_dim=None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # TextEncoder has already consumed the token mask when it pools a
        # caption into one vector. ProjectionHead must explicitly handle the
        # inherited mask instead of letting Keras warn that it is discarded.
        self.supports_masking = True

        if input_dim < 1:
            raise ValueError("input_dim must be positive")

        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        if hidden_dim is None:
            hidden_dim = input_dim

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.projection = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim,
                    name="projection_hidden",
                ),
                tf.keras.layers.Activation(
                    "gelu",
                    name="projection_gelu",
                ),
                tf.keras.layers.Dropout(
                    dropout,
                    name="projection_dropout",
                ),
                tf.keras.layers.Dense(
                    embedding_dim,
                    name="projection_output",
                ),
            ],
            name="projection_network",
        )

        self.output_dim = embedding_dim

    def compute_mask(self, inputs, previous_mask=None):
        # Preserve masks only for sequence-shaped features. A pooled [B,D]
        # embedding has no time axis, so a token mask must not propagate.
        if inputs.shape.rank == 3:
            return previous_mask
        return None

    def call(self, features, training=None):
        features = tf.convert_to_tensor(features)

        tf.debugging.assert_equal(
            tf.shape(features)[-1],
            self.input_dim,
            message="Input feature dimension does not match input_dim",
        )

        embeddings = self.projection(
            features,
            training=training,
        )

        embeddings = tf.math.l2_normalize(
            embeddings,
            axis=-1,
            epsilon=1e-12,
        )

        return embeddings


class CrossModalProjection(tf.keras.layers.Layer):
    """
    Separate projection heads for image and text features.

    Both modalities are mapped to the same embedding dimension.
    """

    def __init__(
        self,
        image_dim=2048,
        text_dim=512,
        embedding_dim=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_projection = ProjectionHead(
            input_dim=image_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            name="image_projection",
        )

        self.text_projection = ProjectionHead(
            input_dim=text_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            name="text_projection",
        )

        self.embedding_dim = embedding_dim

    def call(
        self,
        image_features,
        text_features,
        training=None,
    ):
        image_embeddings = self.image_projection(
            image_features,
            training=training,
        )

        text_embeddings = self.text_projection(
            text_features,
            training=training,
        )

        return image_embeddings, text_embeddings


def _run_projection_tests():
    tf.random.set_seed(42)

    batch_size = 8
    embedding_dim = 512

    print("Cross-Modal Projection Tests")
    print("-" * 55)

    configurations = [
        (2048, 512),
        (2048, 1024),
        (2048, 512),
    ]

    for image_dim, text_dim in configurations:
        model = CrossModalProjection(
            image_dim=image_dim,
            text_dim=text_dim,
            embedding_dim=embedding_dim,
        )

        image_features = tf.random.normal(
            (batch_size, image_dim)
        )

        text_features = tf.random.normal(
            (batch_size, text_dim)
        )

        image_embeddings, text_embeddings = model(
            image_features,
            text_features,
            training=False,
        )

        assert tuple(image_embeddings.shape) == (
            batch_size,
            embedding_dim,
        )

        assert tuple(text_embeddings.shape) == (
            batch_size,
            embedding_dim,
        )

        image_norms = tf.norm(
            image_embeddings,
            ord=2,
            axis=-1,
        )

        text_norms = tf.norm(
            text_embeddings,
            ord=2,
            axis=-1,
        )

        tf.debugging.assert_near(
            image_norms,
            tf.ones((batch_size,)),
            atol=1e-5,
        )

        tf.debugging.assert_near(
            text_norms,
            tf.ones((batch_size,)),
            atol=1e-5,
        )

        similarity = tf.matmul(
            image_embeddings,
            text_embeddings,
            transpose_b=True,
        )

        assert tuple(similarity.shape) == (
            batch_size,
            batch_size,
        )

        print(
            f"Image: {image_dim} -> {embedding_dim} | "
            f"Text: {text_dim} -> {embedding_dim} | PASS"
        )

        print(
            f"Similarity Matrix: {tuple(similarity.shape)}"
        )

    # Verify that the projection head also supports sequence features.
    sequence_projection = ProjectionHead(
        input_dim=512,
        embedding_dim=embedding_dim,
    )

    sequence_features = tf.random.normal(
        (batch_size, 34, 512)
    )

    sequence_embeddings = sequence_projection(
        sequence_features,
        training=False,
    )

    assert tuple(sequence_embeddings.shape) == (
        batch_size,
        34,
        embedding_dim,
    )

    sequence_norms = tf.norm(
        sequence_embeddings,
        ord=2,
        axis=-1,
    )

    tf.debugging.assert_near(
        sequence_norms,
        tf.ones((batch_size, 34)),
        atol=1e-5,
    )

    print("Sequence projection: PASS")

    # Verify gradient flow through both projection branches.
    model = CrossModalProjection(
        image_dim=2048,
        text_dim=512,
        embedding_dim=embedding_dim,
    )

    with tf.GradientTape() as tape:
        image_embeddings, text_embeddings = model(
            tf.random.normal((batch_size, 2048)),
            tf.random.normal((batch_size, 512)),
            training=True,
        )

        loss = (
            tf.reduce_mean(tf.square(image_embeddings))
            + tf.reduce_mean(tf.square(text_embeddings))
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables,
    )

    assert gradients
    assert all(gradient is not None for gradient in gradients)

    print("Gradient flow: PASS")
    print("-" * 55)
    print("All Projection Head tests passed!")


if __name__ == "__main__":
    _run_projection_tests()
