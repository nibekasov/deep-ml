import torch
import torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    Must have <= 2048 trainable params.
    Input:  (N, 1, 28, 28)
    Output: (N, 10) logits
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 1) Небольшой сверточный блок
            # conv1: 1 -> 8 каналов, kernel 3x3, padding=1 (сохраняем 28x28)
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True)

            # 2) Второй сверточный блок
            # conv2: 8 -> 16 каналов, опять 3x3, padding=1
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)

            # 3) MaxPool уменьшает 28x28 -> 14x14 после conv1
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # 4) Global Average Pooling: (N,16,H,W) -> (N,16,1,1)
            self.gap = nn.AdaptiveAvgPool2d(1)

            # 5) Небольшая полносвязная голова: 16 -> 10
            self.fc = nn.Linear(16, 10)

            # Активация
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            # x: (N, 1, 28, 28)
            x = self.act(self.conv1(x))   # (N, 8, 28, 28)
            x = self.pool(x)              # (N, 8, 14, 14)
            x = self.act(self.conv2(x))   # (N, 16, 14, 14)
            x = self.gap(x)               # (N, 16, 1, 1)
            x = x.view(x.size(0), -1)     # (N, 16)
            x = self.fc(x)                # (N, 10) logits
            return x

    return TinyNet()
