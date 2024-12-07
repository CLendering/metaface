from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import random
import os
import sys

# add the 'prototypical' directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from prototypical.utils.sampler import PrototypicalBatchSampler
import json


class DemogPairsDataset(Dataset):
    """Dataset loader for DemogPairs demographic face pairs dataset."""

    def __init__(
        self,
        mode: str = "train",
        root: str = "../datasets/demogpairs/DemogPairs",
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
        cache_images: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        force_new_split: bool = False,
    ):
        """Initialize DemogPairs dataset.

        Args:
            mode: Dataset split ('train', 'val', or 'test')
            root: Root directory containing the dataset
            transform: Transformations to apply to images
            target_transform: Transformations to apply to labels
            cache_images: Whether to cache images in memory
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            seed: Random seed for reproducibility
            force_new_split: Whether to force creation of new splits
        """
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.image_cache: Dict[str, Image.Image] = {}

        if not self.root.exists():
            raise RuntimeError(f"Dataset not found at {self.root}")

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Create or load splits
        self.splits_file = self.root / "splits.json"
        if force_new_split or not self.splits_file.exists():
            self._create_splits(train_ratio, val_ratio)

        # Load appropriate split
        self.classes = self._load_split()
        self.all_items = self._find_items()
        self.class_to_idx = self._index_classes()

        # Prepare data
        self.paths, self.targets = self._prepare_data()
        if cache_images:
            self.images = self._cache_images()

    def _create_splits(self, train_ratio: float, val_ratio: float) -> None:
        """Create random splits of the dataset.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        # Get all person folders
        all_persons = [
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        # Randomly shuffle persons
        random.shuffle(all_persons)

        # Calculate split sizes
        n_persons = len(all_persons)
        n_train = int(n_persons * train_ratio)
        n_val = int(n_persons * val_ratio)

        # Create splits
        splits = {
            "train": all_persons[:n_train],
            "val": all_persons[n_train : n_train + n_val],
            "test": all_persons[n_train + n_val :],
        }

        # Save splits
        with open(self.splits_file, "w") as f:
            json.dump(splits, f)

        print(
            f"Created new splits: train={len(splits['train'])} persons, "
            f"val={len(splits['val'])} persons, test={len(splits['test'])} persons"
        )

    def _load_split(self) -> List[str]:
        """Load class names for current split."""
        with open(self.splits_file) as f:
            splits = json.load(f)
        return splits[self.mode]

    def _find_items(self) -> List[Tuple[str, str, str]]:
        """Find all valid image files and their corresponding classes.

        Returns:
            List of tuples (filename, person_id, full_path)
        """
        items = []
        for person_id in self.classes:
            person_dir = self.root / person_id
            if not person_dir.exists():
                continue

            # Support multiple image formats
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in person_dir.glob(ext):
                    items.append((img_path.name, person_id, str(img_path)))

        print(
            f"Found {len(items)} images from {len(self.classes)} persons for {self.mode} split"
        )
        return items

    def _index_classes(self) -> Dict[str, int]:
        """Create mapping from person IDs to indices."""
        return {person_id: idx for idx, person_id in enumerate(self.classes)}

    def _prepare_data(self) -> Tuple[List[str], List[int]]:
        """Prepare paths and targets for all items."""
        paths = []
        targets = []

        for filename, person_id, full_path in self.all_items:
            paths.append(full_path)
            target = self.class_to_idx[person_id]
            if self.target_transform:
                target = self.target_transform(target)
            targets.append(target)

        return paths, targets

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess a single image."""
        if self.cache_images and path in self.image_cache:
            img = self.image_cache[path]
        else:
            img = Image.open(path).convert("RGB")
            if self.cache_images:
                self.image_cache[path] = img

        # Convert to tensor and normalize
        img = transforms.ToTensor()(img)

        if self.transform:
            img = self.transform(img)

        return img

    def _cache_images(self) -> List[torch.Tensor]:
        """Cache all images in memory."""
        return [self._load_image(path) for path in self.paths]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        if self.cache_images:
            img = self.images[idx]
        else:
            img = self._load_image(self.paths[idx])

        return img, self.targets[idx]

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.all_items)

    def get_person_id(self, idx: int) -> str:
        """Get person ID from class index."""
        return self.classes[idx]


if __name__ == "__main__":
    # Create datasets
    train_dataset = DemogPairsDataset(
        mode="train",
        root="../datasets/demogpairs/DemogPairs",
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        cache_images=True,
        force_new_split=True,  # Only use this for first run
    )

    val_dataset = DemogPairsDataset(
        mode="val",
        root="../datasets/demogpairs/DemogPairs",
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=PrototypicalBatchSampler(
            labels=train_dataset.targets,
            classes_per_it=60,
            num_samples=10,
            iterations=100,
        ),
        num_workers=4,
    )
