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
MODEL_SAVE_PATH = 'Conv2d_3channel_example.pth'
FORCE_RETRAIN = False  # Set to True to force retraining even if saved model exists

print(f"Using device: {DEVICE}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print("MODIFIED FOR 3-CHANNEL INPUT (RGB-like)")
print("-" * 50)

# ########################
# CUSTOM DATASET CREATION
# ########################
class RGB_MNIST(torch.utils.data.Dataset):
    """
    Convert grayscale MNIST to 3-channel RGB-like data
    This simulates working with color images instead of grayscale
    """
    def __init__(self, mnist_dataset, conversion_method='replicate'):
        self.mnist_dataset = mnist_dataset
        self.conversion_method = conversion_method
        
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        # Get the original grayscale image and label
        gray_image, label = self.mnist_dataset[idx]
        
        if self.conversion_method == 'replicate':
            # Method 1: Replicate grayscale across 3 channels (R=G=B)
            rgb_image = gray_image.repeat(3, 1, 1)
            
        elif self.conversion_method == 'colored':
            # Method 2: Create artificial color variations
            # This simulates having actual RGB data with different color patterns
            rgb_image = torch.zeros(3, gray_image.shape[1], gray_image.shape[2])
            
            # Red channel: Original image
            rgb_image[0] = gray_image[0]
            
            # Green channel: Slightly modified (simulate different color response)
            rgb_image[1] = gray_image[0] * 0.8 + torch.randn_like(gray_image[0]) * 0.1
            
            # Blue channel: Another variation
            rgb_image[2] = gray_image[0] * 0.6 + torch.randn_like(gray_image[0]) * 0.1
            
            # Clamp to valid range
            rgb_image = torch.clamp(rgb_image, -3, 3)  # Assuming normalized data
            
        else:
            raise ValueError("conversion_method must be 'replicate' or 'colored'")
        
        return rgb_image, label

def show_cuda_mem():
    if torch.cuda.is_available():
        print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
        print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
        print("Max reserved:", torch.cuda.max_memory_reserved() / 1024**2, "MB")

# ========================
# DATA LOADING
# ========================
print("Loading MNIST dataset and converting to 3-channel...")

# Transform: Convert to tensor and normalize (using MNIST normalization for now)
# Note: For true RGB data, you'd use RGB-specific normalization like:
# transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

# Download and load original MNIST data
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, transform=transform)

# Convert to 3-channel RGB-like data
train_dataset = RGB_MNIST(mnist_train, conversion_method='colored')
test_dataset = RGB_MNIST(mnist_test, conversion_method='colored')

# Demonstrate the conversion
if True:  # Set to False to skip demonstration
    print("\nDemonstrating 3-channel conversion:")
    
    # Get a sample
    rgb_image, label = train_dataset[0]
    original_gray, _ = mnist_train[0]
    
    print(f"Original grayscale shape: {original_gray.shape}")
    print(f"Converted RGB shape: {rgb_image.shape}")
    print(f"Label: {label}")
    
    # Visualize the channels
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original grayscale
    axes[0].imshow(original_gray.squeeze().numpy(), cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')
    
    # Individual RGB channels
    channel_names = ['Red', 'Green', 'Blue']
    for i in range(3):
        axes[i+1].imshow(rgb_image[i].numpy(), cmap='gray')
        axes[i+1].set_title(f'{channel_names[i]} Channel')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ========================
# MODEL DEFINITION
# ========================
class ConvNet3Channel(nn.Module):
    def __init__(self):
        super(ConvNet3Channel, self).__init__()
        
        # CHANGED: First convolutional layer now takes 3 input channels (RGB)
        # instead of 1 (grayscale)
        self.conv1 = nn.Conv2d(3, CONV1_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        
        # Second convolutional layer: same as before
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        
        # Max pooling layer (2x2) - reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # Fully connected layers - same as before
        # After 2 conv+pool operations: 28x28 -> 14x14 -> 7x7
        # So we have 32 channels * 7 * 7 = 1568 features
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        
        # OUTPUT: Still 10 classes for digits 0-9 (not 3!)
        # We're not changing the number of output classes
        self.fc2 = nn.Linear(128, 10)
    
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
model = ConvNet3Channel().to(DEVICE)
print(f"\nModel architecture (3-channel input):")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Compare parameter counts
print(f"\nParameter comparison:")
print(f"Conv1 with 1 input channel: 1 × 16 × 3 × 3 + 16 = {1*16*3*3 + 16:,} parameters")
print(f"Conv1 with 3 input channels: 3 × 16 × 3 × 3 + 16 = {3*16*3*3 + 16:,} parameters")
print(f"Difference: {(3*16*3*3 + 16) - (1*16*3*3 + 16):,} additional parameters")

# ========================
# CHECKPOINT LOADING/SAVING
# ========================
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
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
            'dropout_rate': DROPOUT_RATE,
            'input_channels': 3  # NEW: Track that this model expects 3 input channels
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
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
        
        # Check if hyperparameters match
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
    model = ConvNet3Channel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    print(f"\nNo saved model found - training from scratch...")

# ========================
# TRAINING LOOP
# ========================
if should_train:
    print("\nStarting training with 3-channel input...")
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
print("\nVisualizing learned filters for 3-channel input...")

def visualize_filters_3channel(model, layer_name, num_filters_to_show=8):
    """
    Visualize the learned filters from a convolutional layer with 3-channel input.
    """
    # Get the weights from the specified layer
    if layer_name == 'conv1':
        weights = model.conv1.weight.data.cpu()
        title = f'First Conv Layer Filters (3-channel input → {CONV1_CHANNELS} outputs)'
    elif layer_name == 'conv2':
        weights = model.conv2.weight.data.cpu()
        title = f'Second Conv Layer Filters ({CONV1_CHANNELS}-channel input → {CONV2_CHANNELS} outputs)'
    else:
        raise ValueError("layer_name must be 'conv1' or 'conv2'")
    
    print(f"\n{layer_name} weights shape: {weights.shape}")
    print(f"Interpretation: [output_channels, input_channels, height, width]")
    
    num_filters = min(num_filters_to_show, weights.shape[0])
    
    if layer_name == 'conv1':
        # For first layer with 3 input channels, show each channel separately
        fig, axes = plt.subplots(num_filters, 4, figsize=(16, 2*num_filters))
        fig.suptitle(title)
        
        for i in range(num_filters):
            # Show each input channel separately
            for j in range(3):
                filter_img = weights[i, j, :, :].numpy()
                axes[i, j].imshow(filter_img, cmap='gray')
                axes[i, j].set_title(f'Filter {i+1}, Ch {j+1}')
                axes[i, j].axis('off')
            
            # Show combined magnitude across all input channels
            combined_magnitude = torch.norm(weights[i], dim=0).numpy()
            axes[i, 3].imshow(combined_magnitude, cmap='viridis')
            axes[i, 3].set_title(f'Filter {i+1}, Combined')
            axes[i, 3].axis('off')
            
            print(f"Filter {i+1} stats:")
            print(f"  Ch1: min={weights[i,0].min():.3f}, max={weights[i,0].max():.3f}")
            print(f"  Ch2: min={weights[i,1].min():.3f}, max={weights[i,1].max():.3f}")
            print(f"  Ch3: min={weights[i,2].min():.3f}, max={weights[i,2].max():.3f}")
    
    else:
        # For second layer, show magnitude across all input channels
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
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

# Visualize first layer filters (now with 3 input channels)
visualize_filters_3channel(model, 'conv1')

# Visualize second layer filters
visualize_filters_3channel(model, 'conv2')

# ========================
# SHOW SAMPLE PREDICTIONS
# ========================
print("\nShowing sample predictions with 3-channel input...")

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
fig, axes = plt.subplots(3, 8, figsize=(20, 8))
fig.suptitle('Sample Predictions: 3-Channel Input')

for i in range(8):
    # Show individual channels
    for ch in range(3):
        img_channel = images[i, ch].cpu().numpy()
        axes[ch, i].imshow(img_channel, cmap='gray')
        if ch == 0:
            axes[ch, i].set_title(f'Pred: {predictions[i].item()}, True: {labels[i].item()}')
        axes[ch, i].axis('off')
    
    # Print prediction confidence
    probabilities = F.softmax(outputs[i], dim=0)
    confidence = probabilities[predictions[i]].item()
    print(f"Image {i+1}: Predicted {predictions[i].item()} with {confidence:.2%} confidence")

# Label the rows
for ch, label in enumerate(['Red Channel', 'Green Channel', 'Blue Channel']):
    axes[ch, 0].set_ylabel(label, rotation=90, fontsize=12)

plt.tight_layout()
plt.show()

print(f"\nTraining complete! Final test accuracy: {test_accuracy:.2f}%")
print("\nKey differences with 3-channel input:")
print("- Conv1 now processes 3 input channels instead of 1")
print("- Each filter in Conv1 has 3 separate 3x3 kernels (one per input channel)")
print("- Total parameters increased due to additional input channels")
print("- Output still has 10 classes (digit classification)")
print("- Each filter learns to combine information from all 3 input channels")
print("\nFilter visualization shows:")
print("- How each filter responds to different input channels")
print("- Combined magnitude shows overall filter activation pattern")
print("- Network learns to use color information for digit recognition")
