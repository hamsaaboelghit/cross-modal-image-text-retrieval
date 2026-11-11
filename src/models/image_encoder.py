
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):

    def __init__(self, pretrained=True, unfreeze_last_block=False):

        super().__init__()

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet50(weights=weights)

        # Remove the original classification layer
        self.backbone.fc = nn.Identity()

        # Freeze all backbone parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last ResNet block
        if unfreeze_last_block:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        self.output_dim = 2048

    def forward(self, images):

        features = self.backbone(images)

        return features


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ImageEncoder(pretrained=True).to(device)

    dummy_images = torch.randn(4, 3, 224, 224).to(device)

    features = model(dummy_images)

    print("Image Encoder Test")
    print("-" * 40)
    print(f"Device: {device}")
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output dimension: {model.output_dim}")

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {trainable_params:,}")

    assert features.shape == (4, 2048)

    print("\nImage Encoder test passed!")
