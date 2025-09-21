# models/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Simple CNN compatible with high-res images (internally resizes if needed)
    For CIFAR-like input (32x32) or larger images.
    """

    def __init__(self, num_classes=3, input_size=32):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flatten size after pooling for FC layers
        fc_input_size = 64 * (input_size // 4) * (input_size // 4)
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Resize if input is larger than expected
        if x.size(2) != self.input_size or x.size(3) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
