import argparse
import torch

from models import BaselineCNN
from data import get_corrupted_testloader, get_blurred_testloader
from train import evaluate


def load_model(device):
    model = BaselineCNN().to(device)

    model.load_state_dict(
        torch.load(
            "results/baseline_cnn.pth",
            map_location=device
        )
    )

    return model


def evaluate_corruption(
    corruption_type,
    severity,
    batch_size=128,
    num_workers=0
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = load_model(device)

    if corruption_type == "noise":
        loader = get_corrupted_testloader(
            batch_size=batch_size,
            num_workers=num_workers,
            noise_std=severity
        )

    elif corruption_type == "blur":
        loader = get_blurred_testloader(
            batch_size=batch_size,
            num_workers=num_workers,
            sigma=severity
        )

    else:
        raise ValueError(
            f"Unknown corruption type: {corruption_type}"
        )

    loss, acc = evaluate(model, loader, device)

    print(f"Corruption: {corruption_type}")
    print(f"Severity: {severity}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        choices=["noise", "blur"],
        required=True
    )

    parser.add_argument(
        "--severity",
        type=float,
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    args = parser.parse_args()

    evaluate_corruption(
        corruption_type=args.type,
        severity=args.severity,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )