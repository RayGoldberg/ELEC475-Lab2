"""
ELEC 475: Lab 2 - Full AlexNet Architecture
Adapted for 64x64 RGB input with configurable normalization:
    - 'none': No normalization
    - 'lrn' : Local Response Normalization after ReLU in stages 1-2 (original 2012 paper)
    - 'bn'  : Batch Normalization after every conv, before ReLU (modern)

Reference:
    A. Krizhevsky, I. Sutskever, and G. E. Hinton,
    "ImageNet Classification with Deep Convolutional Neural Networks,"
    NeurIPS 2012.
"""
import torch
import torch.nn as nn


class AlexNet64(nn.Module):
    """
    Spatial dimensions through the network:
        Input   : (B, 3, 64, 64)
        Conv1   : 5x5, s=2, p=2  --> (B, 64, 32, 32)
        Pool1   : 3x3, s=2       --> (B, 64, 15, 15)
        Conv2   : 5x5, s=1, p=2  --> (B, 192, 15, 15)
        Pool2   : 3x3, s=2       --> (B, 192, 7, 7)
        Conv3   : 3x3, s=1, p=1  --> (B, 384, 7, 7)
        Conv4   : 3x3, s=1, p=1  --> (B, 384, 7, 7)
        Conv5   : 3x3, s=1, p=1  --> (B, 256, 7, 7)
        Pool3   : 3x3, s=2       --> (B, 256, 3, 3)
        Flatten : 256 * 3 * 3 = 2,304
        FC1 2,304 -> 1,024, FC2 1,024 -> 1,024, FC3 1,024 -> num_classes
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3, norm_type: str = "bn"):
        super(AlexNet64, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.norm_type = norm_type.lower()

        if self.norm_type not in ("none", "lrn", "bn"):
            raise ValueError(f"Unknown norm_type '{norm_type}'. Choose from 'none', 'lrn', 'bn'.")

        def norm_before_relu(num_features: int) -> nn.Module:
            # Modern block: Conv -> BatchNorm2d -> ReLU (used in all 5 stages when norm_type == "bn")
            return nn.BatchNorm2d(num_features) if self.norm_type == "bn" else nn.Identity()

        def lrn_after_relu() -> nn.Module:
            # 2012 block: Conv -> ReLU -> LRN -> MaxPool (conv1 and conv2 only, as in the original paper)
            # Original AlexNet LRN settings: n=5, alpha=1e-4, beta=0.75, k=2
            if self.norm_type == "lrn":
                return nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
            return nn.Identity()

        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            norm_before_relu(64),
            nn.ReLU(inplace=True),
            lrn_after_relu(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Stage 2
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            norm_before_relu(192),
            nn.ReLU(inplace=True),
            lrn_after_relu(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Stage 3
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            norm_before_relu(384),
            nn.ReLU(inplace=True),
            # Stage 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            norm_before_relu(384),
            nn.ReLU(inplace=True),
            # Stage 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            norm_before_relu(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    from torchinfo import summary
    model = AlexNet64(num_classes=10, in_channels=3, norm_type="bn")
    print("Testing AlexNet-64 Architecture:")
    summary(model, input_size=(1, 3, 64, 64))
