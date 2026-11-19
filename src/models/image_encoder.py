
import tensorflow as tf


class ImageEncoder(tf.keras.Model):
    """
    ResNet50 image encoder.

    Input:
        RGB images with shape [B, H, W, 3].
        Pixel values are expected in the range [0, 255].

    Outputs:
        Global features:  [B, 2048]
        Spatial features: [B, 49, 2048] when return_spatial=True
    """

    def __init__(
        self,
        pretrained=True,
        unfreeze_last_block=False,
        train_all_backbone=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        weights = "imagenet" if pretrained else None

        self.backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=(224, 224, 3),
        )

        self.output_dim = 2048
        self.spatial_dim = 2048

        if unfreeze_last_block and train_all_backbone:
            raise ValueError(
                "Choose either unfreeze_last_block or train_all_backbone, not both."
            )

        # Default architecture-comparison runs keep ResNet frozen.
        self.backbone.trainable = False

        # Full fine-tuning is used for the controlled pretrained-vs-scratch
        # initialization comparison. BatchNorm remains frozen for stability.
        if train_all_backbone:
            self.backbone.trainable = True
            for layer in self.backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        # Optionally fine-tune the final ResNet stage.
        if unfreeze_last_block:
            self.backbone.trainable = True

            for layer in self.backbone.layers:
                if layer.name.startswith("conv5_"):
                    # Keep BatchNormalization frozen.
                    if isinstance(
                        layer,
                        tf.keras.layers.BatchNormalization,
                    ):
                        layer.trainable = False
                    else:
                        layer.trainable = True
                else:
                    layer.trainable = False

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, images, training=False, return_spatial=False):
        images = tf.convert_to_tensor(images)

        tf.debugging.assert_rank(
            images,
            4,
            message="Images must have shape [B, H, W, 3].",
        )

        tf.debugging.assert_equal(
            tf.shape(images)[-1],
            3,
            message="Images must have 3 RGB channels.",
        )

        tf.debugging.assert_positive(
            tf.shape(images)[0],
            message="Image batch cannot be empty.",
        )

        images = tf.cast(images, tf.float32)

        # ResNet50 preprocessing:
        # RGB [0, 255] -> BGR with ImageNet channel centering.
        images = tf.keras.applications.resnet50.preprocess_input(
            images
        )

        # Keep BatchNormalization in inference mode.
        feature_map = self.backbone(
            images,
            training=False,
        )

        global_features = self.global_pool(feature_map)

        if not return_spatial:
            return global_features

        # [B, 7, 7, 2048] -> [B, 49, 2048]
        spatial_features = tf.reshape(
            feature_map,
            [
                tf.shape(feature_map)[0],
                -1,
                self.spatial_dim,
            ],
        )

        return global_features, spatial_features


if __name__ == "__main__":
    print("Image Encoder Tests")
    print("-" * 45)

    # Use weights=None to avoid downloading ImageNet weights
    # during the smoke test.
    model = ImageEncoder(
        pretrained=False,
        unfreeze_last_block=False,
    )

    dummy_images = tf.random.uniform(
        shape=(2, 224, 224, 3),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )

    # Test 1: Global features
    global_features = model(
        dummy_images,
        training=False,
    )

    assert tuple(global_features.shape) == (2, 2048)
    tf.debugging.assert_all_finite(
        global_features,
        "Global features contain NaN or Inf.",
    )

    print("Global features [2, 2048]: PASS")

    # Test 2: Spatial features
    global_features_2, spatial_features = model(
        dummy_images,
        training=False,
        return_spatial=True,
    )

    assert tuple(global_features_2.shape) == (2, 2048)
    assert tuple(spatial_features.shape) == (2, 49, 2048)

    tf.debugging.assert_all_finite(
        spatial_features,
        "Spatial features contain NaN or Inf.",
    )

    print("Spatial features [2, 49, 2048]: PASS")

    # Test 3: Both outputs should use the same feature map.
    tf.debugging.assert_near(
        global_features,
        global_features_2,
        atol=1e-5,
    )

    print("Global output consistency: PASS")

    # Test 4: Frozen backbone
    assert not any(
        layer.trainable
        for layer in model.backbone.layers
    )

    print("Frozen backbone: PASS")

    # Test 5: Fine-tuning configuration
    finetune_model = ImageEncoder(
        pretrained=False,
        unfreeze_last_block=True,
    )

    trainable_conv5_layers = [
        layer
        for layer in finetune_model.backbone.layers
        if layer.name.startswith("conv5_")
        and layer.trainable
        and not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )
    ]

    assert len(trainable_conv5_layers) > 0

    batch_norm_layers = [
        layer
        for layer in finetune_model.backbone.layers
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )
    ]

    assert all(
        not layer.trainable
        for layer in batch_norm_layers
    )

    print("Final block fine-tuning configuration: PASS")
    print("BatchNormalization remains frozen: PASS")

    # Test 6: Gradient flow through trainable final block
    with tf.GradientTape() as tape:
        features = finetune_model(
            dummy_images,
            training=True,
        )
        loss = tf.reduce_mean(tf.square(features))

    gradients = tape.gradient(
        loss,
        finetune_model.trainable_variables,
    )

    assert len(gradients) > 0
    assert all(gradient is not None for gradient in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "A gradient contains NaN or Inf.",
        )

    print("Fine-tuning gradient flow: PASS")

    print("-" * 45)
    print("All Image Encoder tests passed!")
