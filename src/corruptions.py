import torch


def add_gaussian_noise(x, std=0.1):
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0, 1)