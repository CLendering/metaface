from typing import Iterator, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class PrototypicalBatchSampler(Sampler):
    """Yields batches of indexes for few-shot learning episodes.

    Each batch contains indexes for 'num_samples' examples from each of
    'classes_per_it' randomly selected classes. These samples are divided
    into support and query sets within the episode.
    """

    def __init__(
        self, labels: List[int], classes_per_it: int, num_samples: int, iterations: int
    ) -> None:
        """Initialize the PrototypicalBatchSampler.

        Args:
            labels: List of labels for all samples in the dataset
            classes_per_it: Number of classes to sample per episode
            num_samples: Number of samples per class (support + query)
            iterations: Number of episodes per epoch

        Raises:
            ValueError: If any class has fewer samples than num_samples
        """
        super().__init__(data_source=None)

        self.labels = np.array(labels)
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        # Get unique classes and their counts
        self.classes, self.counts = np.unique(self.labels, return_counts=True)

        if min(self.counts) < num_samples:
            raise ValueError(
                f"Every class must have at least {num_samples} samples. "
                f"Found class with only {min(self.counts)} samples."
            )

        # Convert to torch tensors
        self.classes = torch.LongTensor(self.classes)

        # Initialize index matrix
        self.indexes = self._build_index_matrix()

    def _build_index_matrix(self) -> torch.Tensor:
        """Build matrix of indexes for each class.

        Returns:
            Tensor of shape (num_classes, max_samples_per_class) containing
            indexes of samples for each class
        """
        # Create matrix to store indexes
        max_count = int(max(self.counts))
        index_matrix = np.full((len(self.classes), max_count), np.nan)

        # Track number of samples per class
        self.samples_per_class = torch.zeros_like(self.classes)

        # Fill matrix with sample indexes for each class
        for idx, label in enumerate(self.labels):
            label_idx = (self.classes == label).nonzero().item()
            empty_idx = np.where(np.isnan(index_matrix[label_idx]))[0][0]
            index_matrix[label_idx, empty_idx] = idx
            self.samples_per_class[label_idx] += 1

        return torch.from_numpy(index_matrix)

    def _sample_batch(self) -> torch.Tensor:
        """Sample a batch of indexes for one episode.

        Returns:
            Tensor containing randomly sampled indexes
        """
        batch_size = self.sample_per_class * self.classes_per_it
        batch = torch.LongTensor(batch_size)

        # Randomly select classes for this episode
        selected_classes = self.classes[
            torch.randperm(len(self.classes))[: self.classes_per_it]
        ]

        # Sample indexes for each selected class
        for i, class_idx in enumerate(selected_classes):
            # Get index in classes array
            label_idx = (self.classes == class_idx).nonzero().item()

            # Randomly sample indexes for this class
            start_idx = i * self.sample_per_class
            end_idx = (i + 1) * self.sample_per_class
            sample_idxs = torch.randperm(int(self.samples_per_class[label_idx]))[
                : self.sample_per_class
            ]
            batch[start_idx:end_idx] = self.indexes[label_idx][sample_idxs].long()

        # Shuffle batch
        return batch[torch.randperm(batch_size)]

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over batches of indexes.

        Yields:
            Tensor containing indexes for each episode
        """
        for _ in range(self.iterations):
            yield self._sample_batch()

    def __len__(self) -> int:
        """Get number of episodes per epoch.

        Returns:
            Number of iterations/episodes per epoch
        """
        return self.iterations

    def get_class_stats(self) -> Tuple[int, int, float]:
        """Get statistics about class distribution.

        Returns:
            Tuple containing:
                - Minimum samples in any class
                - Maximum samples in any class
                - Average samples per class
        """
        return (
            int(min(self.counts)),
            int(max(self.counts)),
            float(np.mean(self.counts)),
        )
