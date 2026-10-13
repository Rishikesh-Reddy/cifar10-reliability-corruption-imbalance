import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from models import BaselineCNN


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    return loss, accuracy


def train_model(epochs=10, batch_size=128, learning_rate=0.001, num_workers=0):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = BaselineCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    torch.save(model.state_dict(), "results/baseline_cnn.pth")

    print("Model saved to results/baseline_cnn.pth")


if __name__ == "__main__":
    train_model(epochs=10)