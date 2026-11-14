import argparse
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import csv
import os

import matplotlib.pyplot as plt

from data import get_dataloaders, get_mobilenet_dataloaders
from models import BaselineCNN, get_mobilenet


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def evaluate_model(
    model_path,
    model_type="cnn",
    batch_size=128,
    num_workers=0
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if model_type == "mobilenet":
        _, test_loader = get_mobilenet_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        _, test_loader = get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers
        )

    if model_type == "cnn":
        model = BaselineCNN().to(device)

    elif model_type == "mobilenet":
        model = get_mobilenet().to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")

    return (
        all_labels,
        all_predictions,
        accuracy,
        macro_f1
    )

def per_class_accuracy(
    all_labels,
    all_predictions
):
    correct = [0] * 10
    total = [0] * 10

    for label, prediction in zip(
        all_labels,
        all_predictions
    ):
        total[label] += 1

        if label == prediction:
            correct[label] += 1

    accuracies = []

    for i in range(10):
        accuracy = 100 * correct[i] / total[i]
        accuracies.append(accuracy)

        print(
            f"{CLASS_NAMES[i]}: "
            f"{accuracy:.2f}%"
        )

    return accuracies

def save_confusion_matrix(
    labels,
    predictions,
    output_path
):
    cm = confusion_matrix(
        labels,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_metrics(
    output_path,
    model_name,
    accuracy,
    macro_f1
):
    file_exists = os.path.exists(output_path)

    with open(output_path, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "model",
                "accuracy",
                "macro_f1"
            ])

        writer.writerow([
            model_name,
            f"{accuracy * 100:.2f}",
            f"{macro_f1:.4f}"
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--confusion-matrix",
        default=None
    )

    parser.add_argument(
        "--name",
        default="model"
    )

    parser.add_argument(
        "--save-metrics",
        default=None
    )

    parser.add_argument(
        "--model-type",
        choices=["cnn", "mobilenet"],
        default="cnn"
    )


    args = parser.parse_args()

    labels, predictions, accuracy, macro_f1 = evaluate_model(
        model_path=args.model,
        model_type=args.model_type,
        num_workers=args.num_workers
    )


    print("\nPer-class accuracy:")

    per_class_accuracy(
        labels,
        predictions
    )

    if args.confusion_matrix:
        save_confusion_matrix(
            labels,
            predictions,
            args.confusion_matrix
        )

    if args.save_metrics:
        save_metrics(
            args.save_metrics,
            args.name,
            accuracy,
            macro_f1
        )