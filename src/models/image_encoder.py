
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True,
        unfreeze_last_block=False,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet50(weights=weights)

        # Remove the original classification layer.
        self.backbone.fc = nn.Identity()

        # Freeze all backbone parameters initially.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Optionally fine-tune the last ResNet block.
        if unfreeze_last_block:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        self.output_dim = 2048
        self.spatial_dim = 2048

    def train(self, mode=True):
        """
        Keep BatchNorm running statistics frozen.

        Trainable parameters in layer4 can still
        receive gradients when fine-tuning is enabled.
        """
        super().train(mode)

        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        return self

    def forward(
        self,
        images,
        return_spatial=False,
    ):

        if images.ndim != 4:
            raise ValueError(
                "Images must have shape [B, 3, H, W]."
            )

        if images.size(1) != 3:
            raise ValueError(
                "Images must have 3 RGB channels."
            )

        # ResNet50 convolutional feature extraction.
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Spatial feature map: [B, 2048, H', W']
        spatial_map = x

        # Global average pooling: [B, 2048]
        global_features = self.backbone.avgpool(
            spatial_map
        )

        global_features = torch.flatten(
            global_features,
            start_dim=1,
        )

        # Preserve the original behavior by default.
        if not return_spatial:
            return global_features

        # Convert spatial map into a sequence of regions.
        # [B, 2048, 7, 7] -> [B, 49, 2048]
        spatial_features = spatial_map.flatten(
            start_dim=2
        ).transpose(1, 2)

        return global_features, spatial_features


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Image Encoder Tests")
    print("-" * 45)
    print(f"Device: {device}")

    # --------------------------------------------------
    # Test 1: Frozen Backbone
    # --------------------------------------------------

    model = ImageEncoder(
        pretrained=False,
        unfreeze_last_block=False,
    ).to(device)

    model.eval()

    dummy_images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():

        # Original behavior.
        global_features = model(dummy_images)

        # New spatial output.
        global_features_2, spatial_features = model(
            dummy_images,
            return_spatial=True,
        )

    assert global_features.shape == (2, 2048)

    assert global_features_2.shape == (2, 2048)

    assert spatial_features.shape == (2, 49, 2048)

    assert torch.allclose(
        global_features,
        global_features_2,
        atol=1e-6,
    )

    assert torch.isfinite(global_features).all()
    assert torch.isfinite(spatial_features).all()

    print(
        f"Global Features: {list(global_features.shape)} | PASS"
    )

    print(
        f"Spatial Features: {list(spatial_features.shape)} | PASS"
    )

    print("Original output compatibility: PASS")

    # --------------------------------------------------
    # Test 2: BatchNorm Running Statistics
    # --------------------------------------------------

    model.train()

    batch_norm_layers = [
        module
        for module in model.backbone.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]

    assert len(batch_norm_layers) > 0

    for module in batch_norm_layers:
        assert not module.training

    print("BatchNorm evaluation mode: PASS")

    # Save BatchNorm running statistics.
    running_means_before = [
        module.running_mean.clone()
        for module in batch_norm_layers
    ]

    running_vars_before = [
        module.running_var.clone()
        for module in batch_norm_layers
    ]

    # Forward pass in training mode.
    train_images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    train_features = model(train_images)

    # Verify running statistics did not change.
    for index, module in enumerate(batch_norm_layers):

        assert torch.equal(
            module.running_mean,
            running_means_before[index],
        )

        assert torch.equal(
            module.running_var,
            running_vars_before[index],
        )

    print("BatchNorm running statistics: PASS")

    # --------------------------------------------------
    # Test 3: Layer4 Fine-Tuning
    # --------------------------------------------------

    finetune_model = ImageEncoder(
        pretrained=False,
        unfreeze_last_block=True,
    ).to(device)

    finetune_model.train()

    layer4_parameters = list(
        finetune_model.backbone.layer4.parameters()
    )

    assert all(
        param.requires_grad
        for param in layer4_parameters
    )

    frozen_parameters = [
        param
        for name, param in finetune_model.backbone.named_parameters()
        if not name.startswith("layer4.")
    ]

    assert all(
        not param.requires_grad
        for param in frozen_parameters
    )

    assert all(
        not module.training
        for module in finetune_model.backbone.modules()
        if isinstance(module, nn.BatchNorm2d)
    )

    finetune_images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    finetune_features = finetune_model(finetune_images)

    finetune_loss = finetune_features.mean()

    finetune_loss.backward()

    assert all(
        param.grad is not None
        and torch.isfinite(param.grad).all()
        for param in layer4_parameters
    )

    print("Layer4 fine-tuning gradients: PASS")

    print("\nAll Image Encoder tests passed!")

    del model
    del finetune_model

    if device.type == "cuda":
        torch.cuda.empty_cache()
