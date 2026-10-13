import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_mobilenet(num_classes=10):
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model


if __name__ == "__main__":
    cnn = BaselineCNN()
    mobile = get_mobilenet()

    print("Baseline CNN ready")
    print("MobileNetV2 ready")