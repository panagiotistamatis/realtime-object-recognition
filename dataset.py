"""
Data loading and preparation for the CIFAR-100 object recognition subset.

CIFAR-100 is downloaded automatically via torchvision into ./data on first run
and is never committed to the repository.
"""
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from model import (
    NUM_CLASSES,
    SELECTED_CLASS_INDICES,
    NORM_MEAN,
    NORM_STD,
)

BATCH_SIZE = 128


def prepare_data(batch_size=BATCH_SIZE):
    """
    Load and prepare the CIFAR-100 dataset for training, validation and testing.

    Only the selected classroom/office classes are kept and their labels are
    remapped to the range [0, NUM_CLASSES - 1].
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    # Just normalization for validation and testing
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    # Load the CIFAR-100 dataset (auto-downloads on first run)
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Get all CIFAR-100 class names
    all_classes = train_dataset.classes

    # Get the selected class names using our indices
    selected_classes = [all_classes[i] for i in SELECTED_CLASS_INDICES]

    # Print the selected classes for reference
    print(f"Selected classes for office/classroom demo: {selected_classes}")

    # Filter the dataset to include only the selected classes
    train_indices = []
    train_targets = []
    for idx, (_, target) in enumerate(train_dataset):
        if target in SELECTED_CLASS_INDICES:
            train_indices.append(idx)
            # Remap target to be between 0 and NUM_CLASSES-1
            train_targets.append(SELECTED_CLASS_INDICES.index(target))

    test_indices = []
    test_targets = []
    for idx, (_, target) in enumerate(test_dataset):
        if target in SELECTED_CLASS_INDICES:
            test_indices.append(idx)
            # Remap target to be between 0 and NUM_CLASSES-1
            test_targets.append(SELECTED_CLASS_INDICES.index(target))

    # Create custom datasets with remapped labels
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, new_targets):
            self.dataset = dataset
            self.indices = indices
            self.new_targets = new_targets

        def __getitem__(self, idx):
            img, _ = self.dataset[self.indices[idx]]
            return img, self.new_targets[idx]

        def __len__(self):
            return len(self.indices)

    # Create datasets with remapped labels
    train_dataset = RemappedDataset(train_dataset, train_indices, train_targets)
    test_dataset = RemappedDataset(test_dataset, test_indices, test_targets)

    # Split the training data into training and validation sets (80% / 20%)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    # Create data loaders with drop_last=False to handle partial batches correctly
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    return train_loader, val_loader, test_loader, selected_classes


def calculate_class_weights(train_loader, device):
    """
    Calculate class weights to address class imbalance.
    Classes with fewer samples get higher weights.
    """
    class_counts = torch.zeros(NUM_CLASSES)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1

    # Compute weights as inverse of frequency with scaling
    class_weights = 1.0 / class_counts
    class_weights = torch.pow(class_weights, 0.5)  # Square root to reduce extremes
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    print("Class weights:")
    for i, weight in enumerate(class_weights):
        print(f"Class {i}: {weight:.4f}")

    return class_weights.to(device)
