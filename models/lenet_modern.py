"""
ELEC 475: Lab 2 - Modernized LeNet Architecture
Ablation Study 1 & 2: Incorporating ReLU and Overlapping Max Pooling.

Key Modifications from Base LeNet-5:
    1. Tanh -> ReLU
    2. 2x2 Average Pooling -> 3x3 Overlapping Max Pooling with stride=2
"""
import torch
import torch.nn as nn


class LeNetModern(nn.Module):
    """
    Spatial dimension arithmetic:
        Input   : (B, 3, 64, 64)
        Conv1   : 5x5, s=1, p=2  -> (B, 6, 64, 64)
        MaxPool1: 3x3, s=2       -> (B, 6, 31, 31)
        Conv2   : 5x5, s=1, p=2  -> (B, 16, 31, 31)
        MaxPool2: 3x3, s=2       -> (B, 16, 15, 15)
        Flatten : 16 * 15 * 15 = 3,600
        FC1 3,600 -> 120, FC2 120 -> 84, FC3 84 -> num_classes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(LeNetModern, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 15 * 15, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    from torchinfo import summary
    model = LeNetModern(num_classes=10, in_channels=3)
    print("Testing Modernized LeNet Architecture:")
    summary(model, input_size=(1, 3, 64, 64))
