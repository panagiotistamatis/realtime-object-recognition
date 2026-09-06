"""
Model definition and shared configuration for the object recognition project.

A from-scratch CNN with ResNet-style skip connections that classifies 10
everyday classroom/office objects drawn from CIFAR-100.
"""
import torch.nn as nn
import torch.nn.functional as F

# Number of target classes (subset of CIFAR-100)
NUM_CLASSES = 10

# CIFAR-100 image size (32x32 RGB)
IMAGE_SIZE = 32

# CIFAR-100 indices of the selected classroom/office objects.
# These are common objects that are easy to demonstrate in a live setting.
SELECTED_CLASS_INDICES = [
    0,   # apple
    9,   # bottle
    16,  # can
    20,  # chair
    28,  # cup
    39,  # keyboard
    51,  # mushroom
    53,  # orange
    61,  # plate
    82,  # sunflower
]

# Human-readable class names, in the same order as SELECTED_CLASS_INDICES.
# Used by the webcam demo so it does not need to download the dataset just to
# recover the label names.
CLASS_NAMES = [
    "apple",
    "bottle",
    "can",
    "chair",
    "cup",
    "keyboard",
    "mushroom",
    "orange",
    "plate",
    "sunflower",
]

# CIFAR-100 channel normalization statistics.
NORM_MEAN = (0.5071, 0.4866, 0.4409)
NORM_STD = (0.2673, 0.2564, 0.2762)


class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture:
    - Deeper network with BatchNorm and Dropout for better generalization.
    - Skip connections (ResNet-style) for improved gradient flow.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(ImprovedCNN, self).__init__()

        # First block
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        # Second block
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Third block
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Calculate feature size: 32x32 -> 16x16 -> 8x8 -> 4x4
        feature_size = 4 * 4 * 256

        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1 with skip connection
        residual = self.conv1_1(x)
        residual = self.bn1_1(residual)
        residual = F.relu(residual)

        x = self.conv1_2(residual)
        x = self.bn1_2(x)
        x = F.relu(x + residual)  # Skip connection
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2 with skip connection
        residual = self.conv2_1(x)
        residual = self.bn2_1(residual)
        residual = F.relu(residual)

        x = self.conv2_2(residual)
        x = self.bn2_2(x)
        x = F.relu(x + residual)  # Skip connection
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3 with skip connection
        residual = self.conv3_1(x)
        residual = self.bn3_1(residual)
        residual = F.relu(residual)

        x = self.conv3_2(residual)
        x = self.bn3_2(x)
        x = F.relu(x + residual)  # Skip connection
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        return x
