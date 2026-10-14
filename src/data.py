import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np

def get_dataloaders(batch_size=128, data_dir="./data", num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def get_mobilenet_dataloaders(
    batch_size=128,
    data_dir="./data",
    num_workers=0
):
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def get_corrupted_testloader(
    batch_size=128,
    data_dir="./data",
    num_workers=0,
    noise_std=0.1
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.clamp(
                x + torch.randn_like(x) * noise_std,
                0,
                1
            )
        ),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

def get_mobilenet_corrupted_testloader(
    batch_size=128,
    data_dir="./data",
    num_workers=0,
    noise_std=0.1
):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: torch.clamp(
                x + torch.randn_like(x) * noise_std,
                0,
                1
            )
        ),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def get_imbalanced_dataloader(
    batch_size=128,
    data_dir="./data",
    num_workers=0,
    minority_fraction=0.2
):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    targets = np.array(dataset.targets)
    selected_indices = []

    for class_id in range(10):
        class_indices = np.where(targets == class_id)[0]

        if class_id < 5:
            keep = len(class_indices)
        else:
            keep = int(len(class_indices) * minority_fraction)

        selected_indices.extend(class_indices[:keep])

    subset = Subset(dataset, selected_indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)