import torch

from models import BaselineCNN
from data import get_corrupted_testloader
from train import evaluate


def test_noise(std=0.1, batch_size=128, num_workers=0):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = BaselineCNN().to(device)
    model.load_state_dict(
        torch.load(
            "results/baseline_cnn.pth",
            map_location=device
        )
    )

    loader = get_corrupted_testloader(
        batch_size=batch_size,
        num_workers=num_workers,
        noise_std=std
    )

    loss, acc = evaluate(model, loader, device)

    print(f"Noise std: {std}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    for noice in [0.0, 0.05, 0.1, 0.2]:
      print(f"Gaussian Noice: {noice}")
      test_noise(std=noice, num_workers=4)