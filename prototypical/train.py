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
from models.proto_net import ProtoNet
from datasets.bupt_cbface import BUPTCBFaceDataset
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


def init_dataloader(opt, dataset_class, root, mode, force_new_split=False):
    """Initialize data loader."""
    dataset = dataset_class(
        mode=mode,
        root=root,
        transform=get_transforms(),
        cache_images=False,  # Avoid OOM errors
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

    sampler = PrototypicalBatchSampler(
        labels=dataset.targets,
        classes_per_it=(
            opt.classes_per_it_tr if mode == "train" else opt.classes_per_it_val
        ),
        num_samples=(
            opt.num_support_tr + opt.num_query_tr
            if mode == "train"
            else opt.num_support_val + opt.num_query_val
        ),
        iterations=opt.iterations,
    )

    return DataLoader(dataset, batch_sampler=sampler, num_workers=opt.num_workers)


def init_model(opt):
    """Initialize the ProtoNet."""
    device = "cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu"
    model = ProtoNet(x_dim=3, hid_dim=64, z_dim=64)
    return model.to(device)


def train_multi_task(opt, dataloaders, model, optim, lr_scheduler):
    """Train the model with multiple tasks."""
    device = "cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu"
    best_states = {}
    metrics = {
        task: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        for task in dataloaders.keys()
    }
    best_accs = {task: 0 for task in dataloaders.keys()}

    experiment_root = Path(opt.experiment_root)
    # make a run specific directory
    experiment_root = experiment_root / f"run_{len(list(experiment_root.iterdir()))}"
    experiment_root.mkdir(exist_ok=True, parents=True)

    for epoch in range(opt.epochs):
        print(f"=== Epoch: {epoch} ===")

        for task, (tr_dataloader, val_dataloader) in dataloaders.items():
            print(f"Task: {task}")

            # Training
            model.train()
            for batch in tqdm(tr_dataloader, desc=f"Training {task}"):
                optim.zero_grad()
                x, y = [t.to(device) for t in batch]
                model_output = model(x)
                loss, acc = loss_fn(
                    model_output, target=y, n_support=opt.num_support_tr
                )
                loss.backward()
                optim.step()
                metrics[task]["train_loss"].append(loss.item())
                metrics[task]["train_acc"].append(acc.item())

            avg_loss = np.mean(metrics[task]["train_loss"][-opt.iterations :])
            avg_acc = np.mean(metrics[task]["train_acc"][-opt.iterations :])
            print(f"Avg Train Loss: {avg_loss:.4f}, Avg Train Acc: {avg_acc:.4f}")

            # Validation
            if val_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        x, y = [t.to(device) for t in batch]
                        model_output = model(x)
                        loss, acc = loss_fn(
                            model_output, target=y, n_support=opt.num_support_val
                        )
                        metrics[task]["val_loss"].append(loss.item())
                        metrics[task]["val_acc"].append(acc.item())

                avg_loss = np.mean(metrics[task]["val_loss"][-opt.iterations :])
                avg_acc = np.mean(metrics[task]["val_acc"][-opt.iterations :])

                if avg_acc >= best_accs[task]:
                    best_states[task] = model.state_dict()
                    best_accs[task] = avg_acc
                    torch.save(
                        model.state_dict(), experiment_root / f"best_model_{task}.pth"
                    )

        lr_scheduler.step()

    return best_states, best_accs, metrics


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
    options = parse_args()
    init_seed(options)

    # Dictionary of datasets and their configurations
    dataset_configs = {
        "demog_pairs": {
            "dataset_class": DemogPairsDataset,
            "root": "../datasets/demogpairs/DemogPairs",
        },
        "bupt_cbface": {
            "dataset_class": BUPTCBFaceDataset,
            "root": "../datasets/bupt_cbface/BUPT-CBFace-12",
        },
    }

    # Initialize dataloaders for each task
    dataloaders = {}
    for task_name, config in dataset_configs.items():
        dataloaders[task_name] = (
            init_dataloader(
                options, config["dataset_class"], config["root"], "train", True
            ),
            init_dataloader(options, config["dataset_class"], config["root"], "val"),
        )

    model = init_model(options)
    optim = torch.optim.Adam(params=model.parameters(), lr=options.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        gamma=options.lr_scheduler_gamma,
        step_size=options.lr_scheduler_step,
    )

    # Train model on multiple tasks
    best_states, best_accs, metrics = train_multi_task(
        opt=options,
        dataloaders=dataloaders,
        model=model,
        optim=optim,
        lr_scheduler=lr_scheduler,
    )

    # Test on each task
    for task_name, config in dataset_configs.items():
        print(f"\nTesting on {task_name}...")
        test_dataloader = init_dataloader(
            options, config["dataset_class"], config["root"], "test"
        )
        model.load_state_dict(best_states[task_name])
        test(opt=options, test_dataloader=test_dataloader, model=model)


if __name__ == "__main__":
    main()
