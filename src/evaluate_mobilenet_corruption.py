import torch

from models import get_mobilenet
from data import get_mobilenet_corrupted_testloader
from train import evaluate


def test_noise(std=0.1, batch_size=128, num_workers=4):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = get_mobilenet().to(device)

    model.load_state_dict(
        torch.load(
            "results/mobilenet_v2.pth",
            map_location=device
        )
    )

    loader = get_mobilenet_corrupted_testloader(
        batch_size=batch_size,
        num_workers=num_workers,
        noise_std=std
    )

    loss, acc = evaluate(model, loader, device)

    print(f"Noise std: {std}")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    for std in [0.0, 0.05, 0.1, 0.2]:
        test_noise(std=std, num_workers=4)