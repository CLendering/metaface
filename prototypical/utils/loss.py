import torch
import torch.nn.functional as F
from torch.nn import Module
from typing import Tuple, List


class PrototypicalLoss(Module):
    """Loss module for prototypical networks.

    Implements the prototypical loss function for few-shot learning.
    """

    def __init__(self, n_support: int):
        super().__init__()
        self.n_support = n_support

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute euclidean distance between two tensors.

    Args:
        x: First tensor of shape (N x D)
        y: Second tensor of shape (M x D)

    Returns:
        Tensor of shape (N x M) containing pairwise squared Euclidean distances

    Raises:
        ValueError: If tensors have incompatible dimensions
    """
    if x.size(-1) != y.size(-1):
        raise ValueError(f"Incompatible dimensions: {x.size(-1)} != {y.size(-1)}")

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(
    input: torch.Tensor, target: torch.Tensor, n_support: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute prototypical loss and accuracy.

    Args:
        input: Model output features of shape (batch_size x feature_dim)
        target: Ground truth labels
        n_support: Number of support samples per class

    Returns:
        Tuple containing:
            - Loss value (scalar tensor)
            - Accuracy value (scalar tensor)
    """
    # Move tensors to CPU for operations not supported on GPU
    target_cpu = target.cpu()
    input_cpu = input.cpu()

    def get_support_indices(c: int) -> torch.Tensor:
        """Get indices of support samples for class c."""
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    # Get support and query indices
    support_indices = list(map(get_support_indices, classes))
    prototypes = torch.stack(
        [input_cpu[idx_list].mean(0) for idx_list in support_indices]
    )

    query_indices = torch.stack(
        [target_cpu.eq(c).nonzero()[n_support:] for c in classes]
    ).view(-1)

    # Compute distances and probabilities
    query_samples = input_cpu[query_indices]
    distances = euclidean_dist(query_samples, prototypes)
    log_probabilities = F.log_softmax(-distances, dim=1).view(n_classes, n_query, -1)

    # Compute loss and accuracy
    target_indices = (
        torch.arange(n_classes)
        .view(n_classes, 1, 1)
        .expand(n_classes, n_query, 1)
        .long()
    )
    loss = -log_probabilities.gather(2, target_indices).squeeze().mean()

    predictions = log_probabilities.max(2)[1]
    accuracy = predictions.eq(target_indices.squeeze(2)).float().mean()

    return loss, accuracy
