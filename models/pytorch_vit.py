import torch
from torchvision.models import VisionTransformer


class VisionTransformerGrayscale(torch.nn.Module):
    """
    Official PyTorch implementation of Vision Transformer with support for Grayscale (1-channel) images.
    Default parameters are for the MNIST dataset.
    """

    def __init__(
        self,
        image_size=28,
        patch_size=7,
        num_layers=6,
        num_heads=6,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=10,
        dropout=0.1,
    ):
        super(VisionTransformerGrayscale, self).__init__()
        # 1x1 conv to transform 1 channel to 3 channels
        self.conv1 = torch.nn.Conv2d(1, 3, 1)
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.vit(x)
        return x
