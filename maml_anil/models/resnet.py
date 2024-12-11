import torch.nn as nn
import os
import sys
from torchvision import models
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from maml_anil.models.maml_model import MAMLModel


class Conv2dWithDropout(nn.Module):
    def __init__(self, conv_module, dropout_p):
        super(Conv2dWithDropout, self).__init__()
        self.conv = conv_module
        self.dropout = nn.Dropout2d(p=dropout_p, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x

def add_dropout_to_conv_layers(module, dropout_p):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, Conv2dWithDropout(child, dropout_p))
        else:
            add_dropout_to_conv_layers(child, dropout_p)

class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size=64, dropout_p=0.2):
        super(FeatureExtractor, self).__init__()
        self.features = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Unfreeze all layers in the encoder
        for param in self.features.parameters():
            param.requires_grad = True
        self.embedding_size = embedding_size

        # Add dropout to convolutional layers without disrupting parameter tracking
        add_dropout_to_conv_layers(self.features, dropout_p)

        num_ftrs = self.features.fc.in_features

        # Replace the final fully connected layer with an embedding layer
        self.features.fc = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x):
        x = self.features(x)
        # Now x is [batch_size, embedding_size, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, embedding_size]
        return x

class Resnet18Model(MAMLModel):
    def __init__(self, embedding_size=64, output_size=5, dropout_p=0.2):
        features = FeatureExtractor(embedding_size, dropout_p)
        super(Resnet18Model, self).__init__(features, embedding_size, output_size)