import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ========================
# HYPERPARAMETERS SECTION
# ========================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Network architecture parameters
CONV1_CHANNELS = 16  # Number of filters in first conv layer
CONV2_CHANNELS = 32  # Number of filters in second conv layer
KERNEL_SIZE = 3      # Size of convolutional kernels (3x3)
DROPOUT_RATE = 0.5   # Dropout probability

# Model checkpoint settings
MODEL_SAVE_PATH = 'Conv2d_example.pth'
FORCE_RETRAIN = False  # Set to True to force retraining even if saved model exists

print(f"Using device: {DEVICE}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print("-" * 50)

#
# Show Memory
#

def show_cuda_mem():
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
    print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("Max reserved:", torch.cuda.max_memory_reserved() / 1024**2, "MB")

# ========================
# DATA LOADING
# ========================
print("Loading MNIST dataset...")

# Transform: Convert to tensor and normalize (mean=0.1307, std=0.3081 are MNIST specific)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# show some stuff for debug understanding
if False:
    image, label = train_dataset[0]
    print(type(image))       # <class 'torch.Tensor'>
    print(image.shape)       # torch.Size([1, 28, 28])
    print(label)             # e.g., 5

    import matplotlib.pyplot as plt

    # Convert tensor to 2D numpy array
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

    sys.exit(0)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Download and load test data
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ========================
# MODEL DEFINITION
# ========================
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First convolutional layer: 1 input channel (grayscale), 16 output channels
        # Think of this like having 16 different "feature detectors"
        self.conv1 = nn.Conv2d(1, CONV1_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        
        # Second convolutional layer: 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        
        # Max pooling layer (2x2) - reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # Fully connected layers
        # After 2 conv+pool operations: 28x28 -> 14x14 -> 7x7
        # So we have 32 channels * 7 * 7 = 1568 features
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9
    
    def forward(self, x):
        # First conv block: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 32 * 7 * 7)  # Reshape to (batch_size, features)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model instance
model = ConvNet().to(DEVICE)
print(f"\nModel architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ========================
# CHECKPOINT LOADING/SAVING
# ========================
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint - like creating a save point in a video game.
    Saves both the model weights and training state information.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'conv1_channels': CONV1_CHANNELS,
            'conv2_channels': CONV2_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'dropout_rate': DROPOUT_RATE
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint - like loading a saved game.
    Returns the saved training information.
    """
    if os.path.exists(filepath):
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Print loaded information
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        
        print(f"Loaded model from epoch {epoch}")
        print(f"Previous training loss: {loss:.4f}")
        print(f"Previous training accuracy: {accuracy:.2f}%")
        
        # Check if hyperparameters match (optional safety check)
        if 'hyperparameters' in checkpoint:
            saved_params = checkpoint['hyperparameters']
            print(f"Saved hyperparameters: {saved_params}")
        
        return True, epoch, loss, accuracy
    else:
        print(f"No checkpoint found at {filepath}")
        return False, 0, 0, 0

# ========================
# TRAINING SETUP
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nOptimizer: Adam with learning rate {LEARNING_RATE}")
print(f"Loss function: CrossEntropyLoss")

# Try to load existing checkpoint
checkpoint_loaded, start_epoch, prev_loss, prev_accuracy = load_checkpoint(model, optimizer, MODEL_SAVE_PATH)

# Decide whether to train or skip training
should_train = not checkpoint_loaded or FORCE_RETRAIN

if checkpoint_loaded and not FORCE_RETRAIN:
    print(f"\nUsing pre-trained model! Set FORCE_RETRAIN=True to retrain from scratch.")
    print(f"Skipping training and proceeding to evaluation...")
elif FORCE_RETRAIN and checkpoint_loaded:
    print(f"\nFORCE_RETRAIN is enabled - retraining from scratch...")
    # Reset the model and optimizer for fresh training
    model = ConvNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    print(f"\nNo saved model found - training from scratch...")

# ========================
# TRAINING LOOP
# ========================
if should_train:
    print("\nStarting training...")
    flag = False
    model.train()
    
    final_loss = 0
    final_accuracy = 0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device (GPU if available)
            data, target = data.to(DEVICE), target.to(DEVICE)
            if flag:
                show_cuda_mem()
                flag = False
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        final_loss = epoch_loss
        final_accuracy = epoch_accuracy
        
        print(f'Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, '
              f'Training Accuracy: {epoch_accuracy:.2f}%')
        print("-" * 50)
    
    # Save the trained model
    print(f"\nTraining completed! Saving model...")
    save_checkpoint(model, optimizer, EPOCHS, final_loss, final_accuracy, MODEL_SAVE_PATH)
    
else:
    print(f"\nSkipping training - using loaded model.")
    print("-" * 50)

# ========================
# TESTING
# ========================
print("Evaluating on test set...")
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# ========================
# VISUALIZING LEARNED FILTERS
# ========================
print("\nVisualizing learned filters...")

def visualize_filters(model, layer_name, num_filters_to_show=8):
    """
    Visualize the learned filters from a convolutional layer.
    Think of each filter as a "template" the network looks for in images.
    """
    # Get the weights from the specified layer
    if layer_name == 'conv1':
        weights = model.conv1.weight.data.cpu()
        title = f'First Conv Layer Filters (detects basic features like edges)'
    elif layer_name == 'conv2':
        weights = model.conv2.weight.data.cpu()
        title = f'Second Conv Layer Filters (detects complex patterns)'
    else:
        raise ValueError("layer_name must be 'conv1' or 'conv2'")
    
    print(f"\n{layer_name} weights shape: {weights.shape}")
    
    # For conv1: shape is [num_filters, input_channels, height, width] = [16, 1, 3, 3]
    # For conv2: shape is [num_filters, input_channels, height, width] = [32, 16, 3, 3]
    
    num_filters = min(num_filters_to_show, weights.shape[0])
    
    if layer_name == 'conv1':
        # For first layer, we can visualize directly since input is grayscale
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(title)
        
        for i in range(num_filters):
            row = i // 4
            col = i % 4
            
            # Get the filter (squeeze to remove channel dimension for conv1)
            filter_img = weights[i, 0, :, :].numpy()
            
            axes[row, col].imshow(filter_img, cmap='gray')
            axes[row, col].set_title(f'Filter {i+1}')
            axes[row, col].axis('off')
            
            print(f"Filter {i+1} stats: min={filter_img.min():.3f}, "
                  f"max={filter_img.max():.3f}, mean={filter_img.mean():.3f}")
    
    else:
        # For second layer, visualize the magnitude across all input channels
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(title)
        
        for i in range(num_filters):
            row = i // 4
            col = i % 4
            
            # Take the magnitude across all input channels
            filter_magnitude = torch.norm(weights[i], dim=0).numpy()
            
            axes[row, col].imshow(filter_magnitude, cmap='viridis')
            axes[row, col].set_title(f'Filter {i+1}')
            axes[row, col].axis('off')
            
            print(f"Filter {i+1} magnitude stats: min={filter_magnitude.min():.3f}, "
                  f"max={filter_magnitude.max():.3f}, mean={filter_magnitude.mean():.3f}")
    
    plt.tight_layout()
    plt.show()

# Visualize first layer filters
visualize_filters(model, 'conv1')

# Visualize second layer filters (showing magnitude across input channels)
visualize_filters(model, 'conv2')

# ========================
# SHOW SAMPLE PREDICTIONS
# ========================
print("\nShowing some sample predictions...")

# Get a batch of test data
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

# Show first 8 images with predictions
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('Sample Predictions vs True Labels')

for i in range(8):
    row = i // 4
    col = i % 4
    
    # Convert image back to displayable format
    img = images[i].cpu().squeeze().numpy()
    
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f'Pred: {predictions[i].item()}, True: {labels[i].item()}')
    axes[row, col].axis('off')
    
    # Print prediction confidence
    probabilities = F.softmax(outputs[i], dim=0)
    confidence = probabilities[predictions[i]].item()
    print(f"Image {i+1}: Predicted {predictions[i].item()} with {confidence:.2%} confidence")

plt.tight_layout()
plt.show()

print(f"\nTraining complete! Final test accuracy: {test_accuracy:.2f}%")
print("\nWhat the filters learned:")
print("- First layer filters detect basic features like edges, corners, and simple patterns")
print("- Second layer filters combine these basic features to detect more complex shapes")
print("- Together, they create a hierarchy of feature detectors that can recognize digits")

