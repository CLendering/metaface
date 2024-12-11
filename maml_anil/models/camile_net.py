import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from maml_anil.models.maml_model import MAMLModel

class EdgeQKVAttention(nn.Module):
    def __init__(self, dim, heads=2):  # Reduced heads=2 for efficiency
        super().__init__()
        assert dim % heads == 0, "dim should be divisible by heads"

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        # Efficient combined QKV - reduces memory access
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)  # 1x1 conv instead of linear

        # Efficient output projection
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.attn_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Efficient QKV transform using 1x1 convs
        qkv = self.qkv(x)  # B, 3*C, H, W
        qkv = qkv.reshape(B, 3, self.heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, heads, N, head_dim

        # Efficient attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Efficient output computation
        x = (attn @ v).transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        return x


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


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    """

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        """
        :param x_dim: Number of channels in the input
        :param hid_dim: Number of channels in the hidden representations
        :param z_dim: Number of channels in the output
        """
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.attention = EdgeQKVAttention(z_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, embedding_size]
        return x

    def test(self):
        x = torch.randn(128, 3, 84, 84)
        print(self.forward(x).shape)

    def load(
        self,
        path,
        map_location=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        """
        Load the model weights from a file.

        :param path: Path to the file containing the model weights.
        :param map_location: Device to map the model weights to. Default is None.
        """
        self.load_state_dict(torch.load(path, map_location=map_location))


class CamileNet(MAMLModel):
    def __init__(self, input_channels=3, hidden_size=64, embedding_size=64, output_size=5):
        features = ProtoNet(input_channels, hidden_size, embedding_size)
        super(CamileNet, self).__init__(features, embedding_size, output_size)