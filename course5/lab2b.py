"""
Softmax Classifier for MNIST Digit Recognition

This script implements a softmax classifier (single-layer neural network) for 
classifying handwritten digits from the MNIST dataset. Think of it like a 
sorting machine that looks at pixel patterns and decides which digit (0-9) 
an image represents.

The model learns by example, adjusting weights for each pixel to identify 
patterns that distinguish different digits. It's like learning to recognize 
handwriting styles - the model discovers which pixel patterns are most 
indicative of each digit.

Requirements:
    - torch >= 1.8.1
    - torchvision >= 0.9.1
    - matplotlib
    - numpy

Author: [Your name]
Date: May 2025
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split


class SoftMaxClassifier(nn.Module):
    """
    Single-layer neural network with softmax output for multi-class classification.
    
    Think of this as a decision-making layer that takes flattened image pixels
    and directly maps them to class probabilities. Each output neuron learns
    to recognize patterns specific to one digit.
    """
    
    def __init__(self, input_size, output_size):
        super(SoftMaxClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        """Forward pass - like asking 'what digit is this?'"""
        return self.linear(x)


def plot_parameters(model, title="Model Parameters"):
    """
    Visualize the learned weights for each digit class.
    
    Each subplot shows what patterns the model has learned to recognize
    for each digit. Red areas indicate positive weights (important for 
    that digit), blue areas indicate negative weights (not characteristic
    of that digit).
    """
    # Move weights to CPU for visualization
    W = model.state_dict()['linear.weight'].data.cpu()
    w_min, w_max = W.min().item(), W.max().item()
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        # Reshape weights back to 28x28 image format
        weight_image = W[i, :].view(28, 28).numpy()
        
        # Display with consistent color scale
        im = ax.imshow(weight_image, vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_title(f'Digit {i}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_sample(data_sample, prediction=None):
    """Display a single MNIST sample with its label."""
    plt.figure(figsize=(4, 4))
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    
    title = f'True label: {data_sample[1]}'
    if prediction is not None:
        title += f'\nPredicted: {prediction}'
    plt.title(title)
    plt.axis('off')
    plt.show()


def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate model accuracy on a dataset.
    
    Returns:
        accuracy (float): Percentage of correct predictions
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                n_epochs=10, device='cpu'):
    """
    Train the softmax classifier.
    
    Think of training as showing the model many examples until it learns
    the patterns. Like teaching someone to recognize handwriting by showing
    them thousands of examples.
    
    Returns:
        train_losses: List of training losses per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    train_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(n_epochs):
        # Training phase - learning from examples
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for images, labels in train_loader:
            # Flatten images from 28x28 to 784 pixels
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            
            # Forward pass - make predictions
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass - learn from mistakes
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        # Validation phase - test what we've learned
        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{n_epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_accuracies


def plot_training_history(train_losses, val_accuracies):
    """
    Plot training loss and validation accuracy over epochs.
    
    This helps visualize how the model improves over time - like watching
    someone's handwriting recognition skills improve with practice.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def show_predictions(model, dataset, n_samples=5, show_correct=True, device='cpu'):
    """
    Display model predictions on sample images.
    
    Args:
        show_correct: If True, show correctly classified samples.
                     If False, show misclassified samples.
    """
    model.eval()
    softmax = nn.Softmax(dim=-1)
    count = 0
    
    title = "Correctly Classified" if show_correct else "Misclassified"
    print(f"\n{title} Samples:")
    print("-" * 40)
    
    with torch.no_grad():
        for image, label in dataset:
            image_flat = image.view(-1, 28 * 28).to(device)
            
            # Get model prediction
            output = model(image_flat)
            probabilities = softmax(output)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Check if prediction matches label
            is_correct = (predicted.item() == label)
            
            if is_correct == show_correct:
                show_sample((image, label), predicted.item())
                print(f"Confidence: {confidence.item():.2%}\n")
                count += 1
                
            if count >= n_samples:
                break


def main():
    """Main training pipeline."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters - think of these as training settings
    input_dim = 28 * 28  # 784 pixels per image
    output_dim = 10      # 10 digit classes (0-9)
    learning_rate = 0.1
    batch_size = 100
    n_epochs = 10
    
    # Load and prepare MNIST dataset
    print("Loading MNIST dataset...")
    
    # Download and transform data
    transform = transforms.ToTensor()
    full_train_dataset = dsets.MNIST(root='./data', train=True, 
                                    download=True, transform=transform)
    test_dataset = dsets.MNIST(root='./data', train=False, 
                              download=True, transform=transform)
    
    # Split training data: 80% train, 20% validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, 
                                            [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders - these feed data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=5000)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=5000)
    
    # Initialize model
    model = SoftMaxClassifier(input_dim, output_dim)
    print(f"\nModel architecture:\n{model}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Standard loss for classification
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Visualize initial random weights
    print("\nInitial model parameters (random):")
    plot_parameters(model, "Initial Random Weights")
    
    # Train the model
    print("\nTraining model...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        n_epochs, device
    )
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Visualize learned weights
    print("\nLearned model parameters:")
    plot_parameters(model, "Learned Weights After Training")
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_accuracy:.2f}%")
    
    # Show some predictions
    show_predictions(model, test_dataset, n_samples=3, show_correct=True, device=device)
    show_predictions(model, test_dataset, n_samples=3, show_correct=False, device=device)


if __name__ == "__main__":
    main()
