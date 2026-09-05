"""
Training and evaluation entry point.

Trains the from-scratch CNN on the CIFAR-100 classroom/office subset, saves the
best weights to disk and writes the training-curve and confusion-matrix plots.

Run with:
    python train.py

This script does NOT open the webcam. Use webcam.py for the real-time demo.
"""
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model import ImprovedCNN, NUM_CLASSES
from dataset import prepare_data, calculate_class_weights

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
LEARNING_RATE = 0.001  # Learning rate
NUM_EPOCHS = 100       # Epochs
MODEL_FILE = 'simple_classroom_model.pth'


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    """
    Train the model with the provided data loaders and hyperparameters.
    """
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early stopping parameters
    early_stopping_patience = 10
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # Update learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_FILE)
            print(f'Model saved with validation accuracy: {best_val_acc:.4f}')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= early_stopping_patience:
            print(f'Early stopping after {epoch+1} epochs without improvement')
            break

    # Plot training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    # Load the best model for returning
    model.load_state_dict(torch.load(MODEL_FILE))
    return model


def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the model on the test set and write the confusion-matrix plot.
    """
    model.eval()
    correct = 0
    total = 0

    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    # Set up confusion matrix
    conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

                # Update confusion matrix
                conf_matrix[label][pred] += 1

    # Overall accuracy
    test_accuracy = correct / total
    print(f'Overall Test Accuracy: {test_accuracy:.4f}')

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            print(f'{i} ({class_names[i]}): {accuracy:.4f}')

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.grid(False)
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.title('Confusion Matrix')

    # Add class labels
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add numbers to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f'{int(conf_matrix[i, j])}',
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return test_accuracy


def main():
    """
    Orchestrate data preparation, training and evaluation.
    Uses a custom CNN with skip connections (no transfer learning).
    """
    # Check if GPU is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Prepare data
    train_loader, val_loader, test_loader, class_names = prepare_data()
    print(f"Using {NUM_CLASSES} classes for office/classroom objects: {class_names}")

    # Initialize model
    model = ImprovedCNN(num_classes=NUM_CLASSES).to(device)
    print("Using custom CNN with skip connections (no transfer learning)")

    # If a compatible pre-trained model already exists, just evaluate it.
    if os.path.exists(MODEL_FILE):
        try:
            model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            print(f"Loaded pre-trained model from {MODEL_FILE}")
            evaluate_model(model, test_loader, class_names)
            print("\nDone. Weights are available at", MODEL_FILE)
            return
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            print("Training a new model...")

    else:
        print(f"No pre-trained model found at {MODEL_FILE}, training a new model...")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=calculate_class_weights(train_loader, device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)

    # Train and evaluate
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)
    evaluate_model(model, test_loader, class_names)

    print("\nTraining complete.")
    print(f"Your model has been saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
