from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BaseMetaDataset(Dataset):
    """Base class for few-shot learning datasets."""

    def __init__(
        self,
        mode: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
    ):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.classes = []
        self.targets = []

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
