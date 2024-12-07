#!/usr/bin/env python3
"""Command-line argument parser for model training configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration settings for model training."""

    dataset_root: Path
    experiment_root: Path
    epochs: int
    learning_rate: float
    lr_scheduler_step: int
    lr_scheduler_gamma: float
    iterations: int
    classes_per_it_tr: int
    num_support_tr: int
    num_query_tr: int
    classes_per_it_val: int
    num_support_val: int
    num_query_val: int
    manual_seed: int
    cuda: bool
    num_workers: int  # Added num_workers

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create configuration from parsed arguments."""
        return cls(
            dataset_root=Path(args.dataset_root),
            experiment_root=Path(args.experiment_root),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_step=args.lr_scheduler_step,
            lr_scheduler_gamma=args.lr_scheduler_gamma,
            iterations=args.iterations,
            classes_per_it_tr=args.classes_per_it_tr,
            num_support_tr=args.num_support_tr,
            num_query_tr=args.num_query_tr,
            classes_per_it_val=args.classes_per_it_val,
            num_support_val=args.num_support_val,
            num_query_val=args.num_query_val,
            manual_seed=args.manual_seed,
            cuda=args.cuda,
            num_workers=args.num_workers,  # Added num_workers
        )


def get_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Few-shot learning model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path configurations
    path_group = parser.add_argument_group("Path configurations")
    path_group.add_argument(
        "-root",
        "--dataset_root",
        type=str,
        default=os.path.join("..", "datasets", "demogpairs", "DemogPairs/"),
        help="path to dataset",
    )
    path_group.add_argument(
        "-exp",
        "--experiment_root",
        type=str,
        default=os.path.join("..", "output"),
        help="root where to store models, losses and accuracies",
    )

    # Training parameters
    training_group = parser.add_argument_group("Training parameters")
    training_group.add_argument(
        "-nep", "--epochs", type=int, default=100, help="number of epochs to train for"
    )
    training_group.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate for the model",
    )

    # Learning rate scheduler
    scheduler_group = parser.add_argument_group("Learning rate scheduler")
    scheduler_group.add_argument(
        "-lrS",
        "--lr_scheduler_step",
        type=int,
        default=20,
        help="StepLR learning rate scheduler step",
    )
    scheduler_group.add_argument(
        "-lrG",
        "--lr_scheduler_gamma",
        type=float,
        default=0.5,
        help="StepLR learning rate scheduler gamma",
    )

    # Episode configurations
    episode_group = parser.add_argument_group("Episode configurations")
    episode_group.add_argument(
        "-its",
        "--iterations",
        type=int,
        default=100,
        help="number of episodes per epoch",
    )

    # Training episode settings
    train_group = parser.add_argument_group("Training episode settings")
    train_group.add_argument(
        "-cTr",
        "--classes_per_it_tr",
        type=int,
        default=60,
        help="number of random classes per episode for training",
    )
    train_group.add_argument(
        "-nsTr",
        "--num_support_tr",
        type=int,
        default=5,
        help="number of samples per class to use as support for training",
    )
    train_group.add_argument(
        "-nqTr",
        "--num_query_tr",
        type=int,
        default=5,
        help="number of samples per class to use as query for training",
    )

    # Validation episode settings
    val_group = parser.add_argument_group("Validation episode settings")
    val_group.add_argument(
        "-cVa",
        "--classes_per_it_val",
        type=int,
        default=5,
        help="number of random classes per episode for validation",
    )
    val_group.add_argument(
        "-nsVa",
        "--num_support_val",
        type=int,
        default=5,
        help="number of samples per class to use as support for validation",
    )
    val_group.add_argument(
        "-nqVa",
        "--num_query_val",
        type=int,
        default=10,
        help="number of samples per class to use as query for validation",
    )

    # Other settings
    other_group = parser.add_argument_group("Other settings")
    other_group.add_argument(
        "-seed",
        "--manual_seed",
        type=int,
        default=7,
        help="input for the manual seeds initializations",
    )
    other_group.add_argument("--cuda", action="store_true", help="enables cuda")
    other_group.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for data loading",
    )  # Added num_workers

    return parser


def parse_args() -> TrainingConfig:
    """Parse command line arguments and create configuration.

    Returns:
        Training configuration object
    """
    parser = get_parser()
    args = parser.parse_args()
    return TrainingConfig.from_args(args)
