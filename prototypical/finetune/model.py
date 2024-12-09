import torch.nn as nn
import torch
from typing import Any


class FaceClassifier(nn.Module):
    """
    A classifier for face recognition tasks.

    Args:
        encoder (nn.Module): The encoder network.
        num_classes (int): The number of output classes.
        weights_pth (str): Path to the pretrained weights.
        pretrained (bool, optional): Whether to load pretrained weights. Default is True.
        freeze (bool, optional): Whether to freeze the encoder weights. Default is True.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        weights_pth: str,
        pretrained: bool = True,
        freeze: bool = True,
        device: Any = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super(FaceClassifier, self).__init__()
        self.encoder = encoder
        # Get encoder output size
        x = torch.randn(2, 3, 84, 84)
        x = self.encoder(x)
        self.encoder_output_size = x.view(x.size(0), -1).size(1)
        # Initialize classifier with 2 linear layers
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_output_size, num_classes // 2),
            nn.ReLU(),
            nn.Linear(num_classes // 2, num_classes),
        )
        # Load weights of the encoder
        if pretrained:
            self.encoder.load_state_dict(
                torch.load(weights_pth, map_location=device, weights_only=True)
            )
            # Freeze encoder
            if freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the class labels for the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted class labels.
        """
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

    def save(self, pth: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            pth (str): Path to the file.
        """
        torch.save(self.state_dict(), pth)
