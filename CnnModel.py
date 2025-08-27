import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 0
            nn.BatchNorm2d(32), # 1
            nn.ReLU(), # 2
            nn.MaxPool2d(2),  # 3 28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 4
            nn.BatchNorm2d(64), # 5
            nn.ReLU(), # 6
            nn.MaxPool2d(2),  # 7 14x14 → 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8
            nn.BatchNorm2d(128), # 9
            nn.ReLU(), # 10
            nn.MaxPool2d(2),  # 11 7x7 → 3x3

            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 12
            nn.BatchNorm2d(256), # 13
            nn.ReLU(), # 14
            nn.AdaptiveAvgPool2d((1, 1)) # 15
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (256, 1, 1) → (256)
            nn.Linear(256, 128), # 
            nn.ReLU(), # 
            nn.Dropout(0.3), # 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x

# Three-layer MLP model
class MLPModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
