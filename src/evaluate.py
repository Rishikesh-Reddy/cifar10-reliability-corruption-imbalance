import torch
from data import get_dataloaders
from models import BaselineCNN


def per_class_accuracy(model_path, batch_size=128, num_workers=0):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = [0] * 10
    total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            for label, pred in zip(labels, preds):
                total[label.item()] += 1
                correct[label.item()] += int(label == pred)

    for i in range(10):
        print(f"Class {i}: {100 * correct[i] / total[i]:.2f}%")


if __name__ == "__main__":
    per_class_accuracy(
        "results/baseline_cnn_imbalanced.pth",
        num_workers=4
    )