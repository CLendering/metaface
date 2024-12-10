import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from maml_anil.models.maml_model import MAMLModel


def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, hidden_size=64, embedding_size=64):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            conv_block(input_channels, hidden_size),
            conv_block(hidden_size, hidden_size),
            conv_block(hidden_size, hidden_size),
            conv_block(hidden_size, embedding_size),
            nn.AdaptiveAvgPool2d((1,1))  # Ensures output is [N, embedding_size, 1, 1]
        )
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_channels = input_channels

    def forward(self, x):
        x = self.features(x)
        # Now x is [batch_size, embedding_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, embedding_size]
        return x

class SimpleCNN(MAMLModel):
    def __init__(self, input_channels=3, hidden_size=64, embedding_size=64, output_size=5):
        features = FeatureExtractor(input_channels, hidden_size, embedding_size)
        super(SimpleCNN, self).__init__(features, embedding_size, output_size)