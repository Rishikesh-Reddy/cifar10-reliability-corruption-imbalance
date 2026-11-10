import argparse
import torch
from sklearn.metrics import accuracy_score, f1_score

from data import get_dataloaders
from models import BaselineCNN


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
    batch_size=128,
    num_workers=0
):
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

    return all_labels, all_predictions

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

    args = parser.parse_args()

    labels, predictions = evaluate_model(
        model_path=args.model,
        num_workers=args.num_workers
    )

    print("\nPer-class accuracy:")

    per_class_accuracy(
        labels,
        predictions
    )