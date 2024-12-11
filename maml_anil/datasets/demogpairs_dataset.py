from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from maml_anil.datasets.base_dataset import BaseMetaDataset, train_transforms, val_transforms
import json


class DemogPairsDataset(BaseMetaDataset):
    """Dataset loader for DemogPairs demographic face pairs dataset."""

    def __init__(
        self,
        mode: str = "train",
        root: str = "../datasets/demogpairs/DemogPairs/DemoigPairs",
        train_transform: Optional[transforms.Compose] = train_transforms,
        val_transform: Optional[transforms.Compose] = val_transforms,
        test_transform: Optional[transforms.Compose] = val_transforms,
        target_transform: Optional[transforms.Compose] = None,
        cache_images: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        force_new_split: bool = False,
    ):
        super().__init__(mode, train_transform, val_transform, test_transform, target_transform)

        self.root = Path(root)
        self.cache_images = cache_images
        self.image_cache: Dict[str, Image.Image] = {}

        if not self.root.exists():
            raise RuntimeError(f"Dataset not found at {self.root}")

        random.seed(seed)
        np.random.seed(seed)

        bookkeeping_path = os.path.join(self.root.parent, 'demogpairs-bookkeeping-' + mode + '.pkl')

        if os.path.exists(bookkeeping_path):
            print(f"Bookkeeping file found at {bookkeeping_path}")
        else:
            print(f"Bookkeeping file not found at {bookkeeping_path}")
            
        self.splits_file = self.root.parent / "splits.json"

        if force_new_split or not self.splits_file.exists():
            self._create_splits(train_ratio, val_ratio)

            # Delete bookkeeping file if it exists
            if os.path.exists(bookkeeping_path):
                os.remove(bookkeeping_path)

        self.classes = self._load_split()
        self.all_items = self._find_items()
        self.class_to_idx = self._index_classes()

        self.paths, self.targets = self._prepare_data()
        if cache_images:
            self.images = self._cache_images()

        self._bookkeeping_path = bookkeeping_path

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

        if self.mode == "train":
            if self.train_transform:
                img = self.train_transform(img)
        elif self.mode == "val":
            if self.val_transform:
                img = self.val_transform(img)
        elif self.mode == "test":
            if self.test_transform:
                img = self.test_transform(img)

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
