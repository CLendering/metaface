import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
from datasets.vggface2_dataset import VGGFace2Dataset
from models.simple_cnn import SimpleCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised training with a pre-trained feature extractor initialization."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for finetuning"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max number of epochs")
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping"
    )
    parser.add_argument("--use_cuda", type=int, default=1, help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--feature_extractor_path",
        type=str,
        default="best_feature_extractor.pth",
        help="Path to the saved feature extractor weights.",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Fine-tune the feature extractor",
        default=True,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/vggface2/data",
        help="Path to the dataset root",
    )
    return parser.parse_args()


def accuracy(predictions, targets):
    preds = predictions.argmax(dim=1)
    return (preds == targets).sum().float() / targets.size(0)


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    loader_bar = tqdm.tqdm(dataloader)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy(outputs, labels).item() * batch_size
            total_samples += batch_size
            loader_bar.set_description(
                f"Loss: {total_loss / total_samples:.4f} | Acc: {total_acc / total_samples:.4f}"
            )
    return total_loss / total_samples, total_acc / total_samples


def train_one_epoch(model, dataloader, device, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    loader_bar = tqdm.tqdm(dataloader)
    for images, labels in loader_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(outputs, labels).item() * batch_size
        total_samples += batch_size

        loader_bar.set_description(
            f"Loss: {total_loss / total_samples:.4f} | Acc: {total_acc / total_samples:.4f}"
        )

    return total_loss / total_samples, total_acc / total_samples


def main():
    args = parse_args()

    # Set random seed and device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    )

    # Load datasets
    train_dataset = VGGFace2Dataset(
        mode="train", root=args.data_root, force_new_split=True
    )
    val_dataset = VGGFace2Dataset(mode="val", root=args.data_root, force_new_split=True)
    test_dataset = VGGFace2Dataset(
        mode="test", root=args.data_root, force_new_split=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Determine the number of classes
    # Assuming dataset.labels range from 0 to num_classes-1
    num_classes = train_dataset.num_classes

    # Load feature extractor and initialize classifier
    # The feature_extractor in SimpleCNN uses an embedding of size 64*4 = 256 by default if set like before
    # Adjust embedding_size if it differs in your saved model
    base_model = SimpleCNN(
        output_size=num_classes, hidden_size=64, embedding_size=64 * 4
    )
    feature_extractor = base_model.features

    if args.fine_tune:
        print(f"Loading feature extractor from {args.feature_extractor_path}")
        feature_extractor.load_state_dict(
            torch.load(args.feature_extractor_path, map_location=device)
        )
        feature_extractor.to(device)

    # Optionally, decide if you want to fine-tune the feature extractor or freeze it
    # To freeze feature extractor:
    # for param in feature_extractor.parameters():
    #     param.requires_grad = False

    classifier = nn.Linear(64 * 4, num_classes).to(device)

    # Combine feature_extractor + classifier into one model
    model = nn.Sequential(feature_extractor, classifier).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, optimizer, loss_fn
        )
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "feature_extractor": feature_extractor.state_dict(),
                "classifier": classifier.state_dict(),
            }
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # Load the best model and evaluate on test set
    if best_state is not None:
        feature_extractor.load_state_dict(best_state["feature_extractor"])
        classifier.load_state_dict(best_state["classifier"])

    test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
