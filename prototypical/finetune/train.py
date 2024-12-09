from model import FaceClassifier
from vggface2 import get_vggface2_loaders
import torch
import torch.nn as nn
import sys

sys.path.append("../")
from models.proto_net import ProtoNet
from typing import Dict, Any
import tqdm


def train_model(
    loaders: Dict[str, Any],
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: torch.device,
    patience: int = 5,
) -> None:
    """Train the model.

    Args:
        loaders: Data loaders
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        epochs: Number of epochs to train for
        device: Device to run the model on
        patience: Number of epochs to wait before early stopping
    """
    model = model.to(device)
    best_acc = 0.0
    counter = 0
    model_save_path = "face_classifier.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm.tqdm(
            loaders["train"], desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(torch.argmax(outputs, dim=1) == labels)

        epoch_loss = running_loss / len(loaders["train"].dataset)
        epoch_acc = running_corrects.double() / len(loaders["train"].dataset)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(loaders["val"], desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                running_corrects += torch.sum(torch.argmax(outputs, dim=1) == labels)

        epoch_loss = running_loss / len(loaders["val"].dataset)
        epoch_acc = running_corrects.double() / len(loaders["val"].dataset)
        print(f"Val Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            counter += 1

        if counter == patience:
            print("Early stopping")
            break

        lr_scheduler.step()


def test_model(
    loader: torch.utils.data.DataLoader, model: nn.Module, device: torch.device
) -> None:
    """Test the model.

    Args:
        loader: Data loader
        model: Model to test
        device: Device to run the model on
    """
    model = model.to(device)
    model.eval()
    corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            corrects += torch.sum(torch.argmax(outputs, dim=1) == labels)

    accuracy = corrects.double() / len(loader.dataset)
    print(f"Test Acc: {accuracy:.4f}")


def main() -> None:
    loaders = get_vggface2_loaders(
        root="../datasets/vggface2/vggface2",
        batch_size=128,
        num_workers=4,
        val_ratio=0.1,
    )

    model = FaceClassifier(
        num_classes=loaders["train"].dataset.num_classes,
        pretrained=True,
        freeze=False,
        encoder=ProtoNet(
            x_dim=3,
            hid_dim=64,
            z_dim=64,
        ),
        weights_pth="../output/best_model.pth",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model(
        loaders=loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=100,
        device=device,
        patience=5,
    )

    model.load_state_dict(torch.load("face_classifier.pt", weights_only=True))
    test_model(loaders["test"], model, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
