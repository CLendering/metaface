from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch import Tensor


class PrototypicalResNet50(nn.Module):
    """
    ResNet50-based encoder for prototypical networks with configurable architecture.

    Features:
    - Configurable embedding dimension
    - Optional L2 normalization
    - Frozen/unfrozen backbone options
    - Configurable projection head architecture
    - Support for different pooling strategies
    """

    def __init__(
        self,
        freeze_backbone: bool,
        embedding_dim: int = 512,
        weights: Optional[ResNet50_Weights] = None,
        normalize: bool = False,
        projection_hidden_dim: Optional[int] = None,
        pooling_type: str = "avg",
    ) -> None:
        """
        Initialize the PrototypicalResNet50 model.

        Args:
            embedding_dim (int): Dimension of the output embedding space
            weights (ResNet50_Weights, optional): Pretrained weights for ResNet50
            normalize (bool): Whether to L2 normalize embeddings
            freeze_backbone (bool): Whether to freeze ResNet50 backbone
            projection_hidden_dim (int, optional): Hidden dimension for projection head
            pooling_type (str): Type of pooling ('avg', 'max', or 'adaptive_avg')
        """
        super().__init__()
        self.normalize = normalize
        self.pooling_type = pooling_type.lower()

        # Initialize ResNet50 backbone
        self.backbone = models.resnet50(weights=weights)

        # Remove the original FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Initialize pooling layer based on specified type
        if self.pooling_type == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        elif self.pooling_type == "adaptive_avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        else:  # default to average pooling
            self.pooling = nn.AvgPool2d(kernel_size=7, stride=1)

        # Configure projection head
        if projection_hidden_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(2048, projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_dim, embedding_dim),
            )
        else:
            self.projection = nn.Linear(2048, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Extract features through backbone
        features = self.backbone(x)

        # Apply pooling
        features = self.pooling(features)

        # Flatten features
        features = torch.flatten(features, 1)

        # Project to embedding space
        embeddings = self.projection(features)

        # Optional L2 normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the dimension of the embedding space."""
        return (
            self.projection[-1].out_features
            if isinstance(self.projection, nn.Sequential)
            else self.projection.out_features
        )

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extract features before the projection head.
        Useful for transfer learning or feature analysis.
        """
        features = self.backbone(x)
        features = self.pooling(features)
        return torch.flatten(features, 1)
