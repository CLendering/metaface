import torch.nn as nn
import torch

def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module

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

class MAMLModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, final_embedding_size=64, output_size=5):
        super(MAMLModel, self).__init__()
        self.features = feature_extractor
        self.classifier = nn.Linear(final_embedding_size, output_size, bias=True)

        maml_init_(self.classifier)
        self.output_size = output_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
