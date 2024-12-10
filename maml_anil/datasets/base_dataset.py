from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(146),
        transforms.RandomCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(146),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet
    ]
)

class BaseMetaDataset(Dataset):
    """Base class for few-shot learning datasets."""

    def __init__(
        self,
        mode: str = "train",
        train_transform: Optional[transforms.Compose] = train_transforms,
        val_transform: Optional[transforms.Compose] = val_transforms,
        test_transform: Optional[transforms.Compose] = val_transforms,
        target_transform: Optional[transforms.Compose] = None,
    ):
        self.mode = mode
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.target_transform = target_transform
        self.test_transform = test_transform
        self.classes = []
        self.targets = []

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError