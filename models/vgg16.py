import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict


class VGG16Encoder(nn.Module):
    """VGG16 encoder for Prototypical Networks.

    Creates embeddings in a 512-dimensional space without any classification layers.
    The network ends with the last convolutional block followed by adaptive pooling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        use_batch_norm: bool = True,
        pretrained: bool = False,
    ):
        """Initialize VGG16 encoder.

        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            feature_dim: Dimension of output feature embeddings
            use_batch_norm: Whether to use batch normalization
            pretrained: Whether to initialize with pre-trained weights
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.features = self._make_features(in_channels, use_batch_norm)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained and in_channels == 3:
            self._load_pretrained_weights()
        else:
            self._initialize_weights()

    def _make_features(self, in_channels: int, use_batch_norm: bool) -> nn.Sequential:
        """Create feature extraction layers.

        Args:
            in_channels: Number of input channels
            use_batch_norm: Whether to use batch normalization

        Returns:
            Sequential container of feature extraction layers
        """
        # VGG16 configuration: numbers represent output channels, 'M' represents maxpool
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        layers: List[tuple[str, nn.Module]] = []
        current_channels = in_channels

        for i, v in enumerate(cfg):
            if v == "M":
                layers.append(
                    (f"maxpool{len(layers)}", nn.MaxPool2d(kernel_size=2, stride=2))
                )
            else:
                v = int(v)
                conv2d = nn.Conv2d(current_channels, v, kernel_size=3, padding=1)
                layers.append((f"conv{len(layers)}", conv2d))

                if use_batch_norm:
                    layers.append((f"bn{len(layers)}", nn.BatchNorm2d(v)))

                layers.append((f"relu{len(layers)}", nn.ReLU(inplace=True)))
                current_channels = v

        return nn.Sequential(OrderedDict(layers))

    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self) -> None:
        """Load pre-trained weights from torchvision model."""
        try:
            import torchvision.models as models

            pretrained_dict = models.vgg16(pretrained=True).state_dict()
            model_dict = self.state_dict()

            # Filter out unnecessary keys (only keep feature layers)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and "features" in k
            }

            # Update model weights
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate embeddings.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Feature embeddings of shape (batch_size, feature_dim)
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x.view(x.size(0), -1)
