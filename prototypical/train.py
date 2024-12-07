#!/usr/bin/env python3
"""Main training script for Prototypical Networks."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.sampler import PrototypicalBatchSampler
from utils.loss import prototypical_loss as loss_fn
from datasets.demogpairs import DemogPairsDataset
from models.vgg16 import VGG16Encoder as ProtoNet
from config import parse_args
from torchvision import transforms


def init_seed(opt):
    """Set random seeds for reproducibility."""
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def get_transforms():
    """Get image transforms."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def init_dataset(opt, mode, force_new_split=False):
    """Initialize dataset."""
    dataset = DemogPairsDataset(
        mode=mode,
        root=opt.dataset_root,
        transform=get_transforms(),
        cache_images=True,
        force_new_split=force_new_split,
    )

    n_classes = len(dataset.classes)
    if mode == "train" and n_classes < opt.classes_per_it_tr:
        raise ValueError(
            f"Not enough training classes ({n_classes}) "
            f"for classes_per_it_tr ({opt.classes_per_it_tr})"
        )
    elif mode == "val" and n_classes < opt.classes_per_it_val:
        raise ValueError(
            f"Not enough validation classes ({n_classes}) "
            f"for classes_per_it_val ({opt.classes_per_it_val})"
        )

    return dataset


def init_sampler(opt, labels, mode):
    """Initialize prototypical batch sampler."""
    if "train" in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(
        labels=labels,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        iterations=opt.iterations,
    )


def init_dataloader(opt, mode, force_new_split=False):
    """Initialize data loader."""
    dataset = init_dataset(opt, mode, force_new_split)
    sampler = init_sampler(opt, dataset.targets, mode)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=opt.num_workers)


def init_model(opt):
    """Initialize the ProtoNet."""
    device = "cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu"
    model = ProtoNet(
        in_channels=3, feature_dim=512, use_batch_norm=True  # RGB images
    ).to(device)
    return model


def init_optim(opt, model):
    """Initialize optimizer."""
    return torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    """Initialize learning rate scheduler."""
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optim, gamma=opt.lr_scheduler_gamma, step_size=opt.lr_scheduler_step
    )


def save_list_to_file(path, thelist):
    """Save list to file."""
    with open(path, "w") as f:
        for item in thelist:
            f.write(f"{item}\n")


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    """Train the model with the prototypical learning algorithm."""
    device = "cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu"
    best_state = None if val_dataloader is None else None

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    best_acc = 0

    experiment_root = Path(opt.experiment_root)
    best_model_path = experiment_root / "best_model.pth"
    last_model_path = experiment_root / "last_model.pth"

    for epoch in range(opt.epochs):
        print(f"=== Epoch: {epoch} ===")

        # Training phase
        model.train()
        for batch in tqdm(tr_dataloader, desc="Training"):
            optim.zero_grad()
            x, y = [t.to(device) for t in batch]
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-opt.iterations :])
        avg_acc = np.mean(train_acc[-opt.iterations :])
        print(f"Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {avg_acc:.4f}")
        lr_scheduler.step()

        # Validation phase
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    x, y = [t.to(device) for t in batch]
                    model_output = model(x)
                    loss, acc = loss_fn(
                        model_output, target=y, n_support=opt.num_support_val
                    )
                    val_loss.append(loss.item())
                    val_acc.append(acc.item())

                avg_loss = np.mean(val_loss[-opt.iterations :])
                avg_acc = np.mean(val_acc[-opt.iterations :])
                postfix = (
                    " (Best)" if avg_acc >= best_acc else f" (Best: {best_acc:.4f})"
                )
                print(
                    f"Avg Val Loss: {avg_loss:.4f}, Avg Val Acc: {avg_acc:.4f}{postfix}"
                )

                if avg_acc >= best_acc:
                    torch.save(model.state_dict(), best_model_path)
                    best_acc = avg_acc
                    best_state = model.state_dict()

    # Save final model
    torch.save(model.state_dict(), last_model_path)

    # Save metrics
    for name in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        save_list_to_file(experiment_root / f"{name}.txt", locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


@torch.no_grad()
def test(opt, test_dataloader, model):
    """Test the model."""
    device = "cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu"
    model.eval()
    accuracies = []

    for _ in range(10):  # 10 test epochs
        for batch in test_dataloader:
            x, y = [t.to(device) for t in batch]
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y, n_support=opt.num_support_val)
            accuracies.append(acc.item())

    avg_acc = np.mean(accuracies)
    print(f"Test Acc: {avg_acc:.4f}")
    return avg_acc


def main():
    """Main training function."""
    # Parse arguments
    options = parse_args()

    # Create experiment directory
    experiment_root = Path(options.experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    # CUDA warning
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: CUDA device available but not used (--cuda not set)")

    # Initialize everything
    init_seed(options)

    # Create dataloaders (force new split only for first run)
    tr_dataloader = init_dataloader(options, "train", force_new_split=True)
    val_dataloader = init_dataloader(options, "val")
    test_dataloader = init_dataloader(options, "test")

    # Initialize model and training components
    model = init_model(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)

    # Train model
    res = train(
        opt=options,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optim=optim,
        lr_scheduler=lr_scheduler,
    )
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res

    # Test both last and best models
    print("Testing with last model...")
    test(opt=options, test_dataloader=test_dataloader, model=model)

    print("Testing with best model...")
    model.load_state_dict(best_state)
    test(opt=options, test_dataloader=test_dataloader, model=model)


if __name__ == "__main__":
    main()
