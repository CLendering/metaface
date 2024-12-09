from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import random
from collections import defaultdict
import json
from typing import Optional, Tuple, List, Dict


class VGGFace2Dataset(Dataset):
    """Dataset loader for VGGFace2."""

    def __init__(
        self,
        mode: str = "train",  # 'train', 'val', or 'test'
        root: str = "datasets/vggface2/vggface2",
        transform: Optional[transforms.Compose] = None,
        val_ratio: float = 0.1,
        seed: int = 42,
        force_new_split: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.transform = transform

        random.seed(seed)
        np.random.seed(seed)

        self.splits_file = self.root / "train_val_splits.json"
        if force_new_split or not self.splits_file.exists():
            self._create_splits(val_ratio)

        self.image_paths, self.targets = self._load_split()
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted(set(self.targets)))
        }
        self.targets = [self.class_to_idx[cls] for cls in self.targets]

        self.num_classes = len(self.class_to_idx)

        print(
            f"Loaded {self.mode} split: {len(self.image_paths)} images, "
            f"{len(self.class_to_idx)} classes"
        )

    def _create_splits(self, val_ratio: float):
        """Create train/val splits from train folder and use val folder for test."""
        splits = {}

        # Handle train and validation splits (from train folder)
        train_folder = self.root / "train"
        identity_folders = sorted(
            [
                d
                for d in train_folder.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

        train_paths = []
        train_labels = []
        val_paths = []
        val_labels = []

        for folder in identity_folders:
            identity = folder.name
            images = list(folder.glob("*.jpg"))
            n_images = len(images)
            n_val = max(int(n_images * val_ratio), 1)  # At least 1 validation sample

            # Randomly shuffle images
            random.shuffle(images)

            # Split images
            val_imgs = images[:n_val]
            train_imgs = images[n_val:]

            # Add to splits
            train_paths.extend([str(img.relative_to(self.root)) for img in train_imgs])
            train_labels.extend([identity] * len(train_imgs))
            val_paths.extend([str(img.relative_to(self.root)) for img in val_imgs])
            val_labels.extend([identity] * len(val_imgs))

        # Handle test split (from val folder)
        test_folder = self.root / "val"
        test_paths = []
        test_labels = []

        for folder in test_folder.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                identity = folder.name
                images = list(folder.glob("*.jpg"))
                test_paths.extend([str(img.relative_to(self.root)) for img in images])
                test_labels.extend([identity] * len(images))

        # Save all splits
        splits = {
            "train": {"paths": train_paths, "labels": train_labels},
            "val": {"paths": val_paths, "labels": val_labels},
            "test": {"paths": test_paths, "labels": test_labels},
        }

        with open(self.splits_file, "w") as f:
            json.dump(splits, f)

        print(
            f"Created splits: train={len(train_paths)}, "
            f"val={len(val_paths)}, test={len(test_paths)} images"
        )

    def _load_split(self) -> Tuple[List[str], List[str]]:
        """Load the appropriate split."""
        with open(self.splits_file) as f:
            splits = json.load(f)
        split_data = splits[self.mode]
        return split_data["paths"], split_data["labels"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        img_path = self.root / self.image_paths[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.image_paths)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get image transforms."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((84, 84)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )


def get_vggface2_loaders(
    root: str, batch_size: int = 32, num_workers: int = 4, val_ratio: float = 0.1
) -> Dict[str, torch.utils.data.DataLoader]:
    """Get train, validation and test dataloaders."""
    # Initialize datasets
    train_dataset = VGGFace2Dataset(
        mode="train",
        root=root,
        transform=get_transforms(train=True),
        val_ratio=val_ratio,
        force_new_split=True,
    )

    val_dataset = VGGFace2Dataset(
        mode="val",
        root=root,
        transform=get_transforms(train=False),
        val_ratio=val_ratio,
    )

    test_dataset = VGGFace2Dataset(
        mode="test",
        root=root,
        transform=get_transforms(train=False),
        val_ratio=val_ratio,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


if __name__ == "__main__":
    # Create dataloaders
    loaders = get_vggface2_loaders(
        root="../datasets/vggface2/vggface2",
        batch_size=32,
        num_workers=4,
        val_ratio=0.1,
    )

    # Check sizes
    for split, loader in loaders.items():
        print(f"{split.capitalize()} loader: {len(loader.dataset)} samples")
