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
EPOCHS = 10  # More epochs for CIFAR-10 (more complex than MNIST)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Network architecture parameters
CONV1_CHANNELS = 32  # More filters for color images
CONV2_CHANNELS = 64  # 
CONV3_CHANNELS = 128 # Added third conv layer for better feature extraction
KERNEL_SIZE = 3      # Size of convolutional kernels (3x3)
DROPOUT_RATE = 0.5   # Dropout probability

# Model checkpoint settings
MODEL_SAVE_PATH = 'CIFAR10_CNN_example.pth'
FORCE_RETRAIN = False  # Set to True to force retraining even if saved model exists

# CIFAR-10 class names
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Using device: {DEVICE}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print("USING CIFAR-10 COLOR DATASET (32x32 RGB images)")
print("-" * 60)

def show_cuda_mem():
    if torch.cuda.is_available():
        print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
        print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
        print("Max reserved:", torch.cuda.max_memory_reserved() / 1024**2, "MB")

# ========================
# DATA LOADING
# ========================
print("Loading CIFAR-10 dataset...")

# CIFAR-10 specific transforms with data augmentation for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.RandomRotation(10),            # Small rotations
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
])

# Test transform (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats  
])

# Download and load CIFAR-10 data
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(CIFAR10_CLASSES)}")
print(f"Classes: {', '.join(CIFAR10_CLASSES)}")

# Show some sample images
def show_cifar10_samples():
    """Display sample CIFAR-10 images"""
    print("\nDisplaying sample CIFAR-10 images...")
    
    # Get a batch of training data (without normalization for visualization)
    vis_transform = transforms.Compose([transforms.ToTensor()])
    vis_dataset = datasets.CIFAR10('./data', train=True, transform=vis_transform)
    vis_loader = DataLoader(vis_dataset, batch_size=16, shuffle=True)
    
    dataiter = iter(vis_loader)
    images, labels = next(dataiter)
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))
    fig.suptitle('Sample CIFAR-10 Images (32x32 RGB)', fontsize=14)
    
    for i in range(16):
        row = i // 8
        col = i % 8
        
        # Convert from tensor to numpy and transpose for matplotlib
        img = images[i].permute(1, 2, 0).numpy()  # CHW -> HWC
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{CIFAR10_CLASSES[labels[i]]}', fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Show samples
show_cifar10_samples()

# ========================
# MODEL DEFINITION
# ========================
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, CONV1_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV1_CHANNELS)  # Batch norm for better training
        
        # Second convolutional block  
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV2_CHANNELS)
        
        # Third convolutional block (added for better feature extraction)
        self.conv3 = nn.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, kernel_size=KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV3_CHANNELS)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # Calculate the size after convolutions and pooling
        # CIFAR-10 images are 32x32
        # After conv1 + pool: 32x32 -> 16x16
        # After conv2 + pool: 16x16 -> 8x8  
        # After conv3 + pool: 8x8 -> 4x4
        # So we have: 128 channels * 4 * 4 = 2048 features
        self.fc1 = nn.Linear(CONV3_CHANNELS * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes in CIFAR-10
    
    def forward(self, x):
        # First conv block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, CONV3_CHANNELS * 4 * 4)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Create model instance
model = CIFAR10_CNN().to(DEVICE)
print(f"\nModel architecture for CIFAR-10:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print(f"\nArchitecture breakdown:")
print(f"Input: 3 channels (RGB) × 32×32 pixels")
print(f"Conv1: 3 → {CONV1_CHANNELS} channels")
print(f"Conv2: {CONV1_CHANNELS} → {CONV2_CHANNELS} channels") 
print(f"Conv3: {CONV2_CHANNELS} → {CONV3_CHANNELS} channels")
print(f"Final feature map: {CONV3_CHANNELS} × 4×4 = {CONV3_CHANNELS * 4 * 4} features")
print(f"Output: 10 classes")

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
            'conv3_channels': CONV3_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'dropout_rate': DROPOUT_RATE,
            'dataset': 'CIFAR-10'
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        
        print(f"Loaded model from epoch {epoch}")
        print(f"Previous training loss: {loss:.4f}")
        print(f"Previous training accuracy: {accuracy:.2f}%")
        
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
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Added weight decay

print(f"\nOptimizer: Adam with learning rate {LEARNING_RATE}")
print(f"Loss function: CrossEntropyLoss")
print(f"Weight decay: 1e-4 (for regularization)")

# Try to load existing checkpoint
checkpoint_loaded, start_epoch, prev_loss, prev_accuracy = load_checkpoint(model, optimizer, MODEL_SAVE_PATH)

should_train = not checkpoint_loaded or FORCE_RETRAIN

if checkpoint_loaded and not FORCE_RETRAIN:
    print(f"\nUsing pre-trained model! Set FORCE_RETRAIN=True to retrain from scratch.")
    print(f"Skipping training and proceeding to evaluation...")
elif FORCE_RETRAIN and checkpoint_loaded:
    print(f"\nFORCE_RETRAIN is enabled - retraining from scratch...")
    model = CIFAR10_CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
else:
    print(f"\nNo saved model found - training from scratch...")

# ========================
# TRAINING LOOP
# ========================
if should_train:
    print("\nStarting training on CIFAR-10...")
    model.train()
    
    final_loss = 0
    final_accuracy = 0
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Update learning rate
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        final_loss = epoch_loss
        final_accuracy = epoch_accuracy
        
        print(f'Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, '
              f'Training Accuracy: {epoch_accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
        print("-" * 60)
    
    print(f"\nTraining completed! Saving model...")
    save_checkpoint(model, optimizer, EPOCHS, final_loss, final_accuracy, MODEL_SAVE_PATH)
    
else:
    print(f"\nSkipping training - using loaded model.")
    print("-" * 60)

# ========================
# TESTING
# ========================
print("Evaluating on CIFAR-10 test set...")
model.eval()

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Per-class accuracy
        c = (predicted == target).squeeze()
        for i in range(target.size(0)):
            label = target[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

test_accuracy = 100 * correct / total
print(f'Overall Test Accuracy: {test_accuracy:.2f}%')

print(f"\nPer-class accuracy:")
for i, class_name in enumerate(CIFAR10_CLASSES):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'{class_name:>12}: {accuracy:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')

# ========================
# VISUALIZE LEARNED FILTERS
# ========================
print("\nVisualizing learned filters for CIFAR-10...")

def visualize_cifar_filters(model, layer_name, num_filters_to_show=16):
    """Visualize filters learned on CIFAR-10"""
    
    if layer_name == 'conv1':
        weights = model.conv1.weight.data.cpu()
        title = f'First Conv Layer Filters (RGB → {CONV1_CHANNELS} channels)'
    elif layer_name == 'conv2':
        weights = model.conv2.weight.data.cpu()
        title = f'Second Conv Layer Filters ({CONV1_CHANNELS} → {CONV2_CHANNELS} channels)'
    elif layer_name == 'conv3':
        weights = model.conv3.weight.data.cpu()
        title = f'Third Conv Layer Filters ({CONV2_CHANNELS} → {CONV3_CHANNELS} channels)'
    else:
        raise ValueError("layer_name must be 'conv1', 'conv2', or 'conv3'")
    
    print(f"\n{layer_name} weights shape: {weights.shape}")
    
    num_filters = min(num_filters_to_show, weights.shape[0])
    
    if layer_name == 'conv1':
        # For RGB input, show each channel and combined
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(title)
        
        for i in range(min(16, num_filters)):
            row = i // 4
            col = i % 4
            
            # For conv1, we can visualize the RGB filter directly
            filter_rgb = weights[i].permute(1, 2, 0).numpy()  # CHW -> HWC
            
            # Normalize for display
            filter_rgb = (filter_rgb - filter_rgb.min()) / (filter_rgb.max() - filter_rgb.min())
            
            axes[row, col].imshow(filter_rgb)
            axes[row, col].set_title(f'Filter {i+1}')
            axes[row, col].axis('off')
            
            print(f"Filter {i+1} - R: [{weights[i,0].min():.3f}, {weights[i,0].max():.3f}], "
                  f"G: [{weights[i,1].min():.3f}, {weights[i,1].max():.3f}], "
                  f"B: [{weights[i,2].min():.3f}, {weights[i,2].max():.3f}]")
    else:
        # For deeper layers, show magnitude
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(title)
        
        for i in range(min(16, num_filters)):
            row = i // 4
            col = i % 4
            
            filter_magnitude = torch.norm(weights[i], dim=0).numpy()
            
            axes[row, col].imshow(filter_magnitude, cmap='viridis')
            axes[row, col].set_title(f'Filter {i+1}')
            axes[row, col].axis('off')
            
            print(f"Filter {i+1} magnitude: min={filter_magnitude.min():.3f}, "
                  f"max={filter_magnitude.max():.3f}, mean={filter_magnitude.mean():.3f}")
    
    plt.tight_layout()
    plt.show()

# Visualize filters from all layers
visualize_cifar_filters(model, 'conv1')
visualize_cifar_filters(model, 'conv2')
visualize_cifar_filters(model, 'conv3')

# ========================
# SHOW SAMPLE PREDICTIONS
# ========================
print("\nShowing sample predictions on CIFAR-10...")

# Get a batch of test data
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

model.eval()
with torch.no_grad():
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)

# Show first 12 images with predictions
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('CIFAR-10 Predictions vs True Labels')

# Denormalize for display
mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(DEVICE)
std = torch.tensor([0.2023, 0.1994, 0.2010]).to(DEVICE)

for i in range(12):
    row = i // 4
    col = i % 4
    
    # Denormalize image
    img = images[i] * std.view(3, 1, 1) + mean.view(3, 1, 1)
    img = torch.clamp(img, 0, 1)
    img = img.cpu().permute(1, 2, 0).numpy()
    
    axes[row, col].imshow(img)
    
    pred_class = CIFAR10_CLASSES[predictions[i]]
    true_class = CIFAR10_CLASSES[labels[i]]
    confidence = probabilities[i][predictions[i]].item()
    
    color = 'green' if predictions[i] == labels[i] else 'red'
    axes[row, col].set_title(f'Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.2%}', 
                            color=color, fontsize=10)
    axes[row, col].axis('off')
    
    print(f"Image {i+1}: Predicted '{pred_class}' (True: '{true_class}') with {confidence:.2%} confidence")

plt.tight_layout()
plt.show()

print(f"\nCIFAR-10 CNN Results Summary:")
print(f"Dataset: 50,000 training + 10,000 test RGB images (32×32)")
print(f"Classes: {len(CIFAR10_CLASSES)} ({', '.join(CIFAR10_CLASSES)})")
print(f"Final test accuracy: {test_accuracy:.2f}%")
print(f"Architecture: 3-layer CNN with BatchNorm and data augmentation")
print(f"Total parameters: {total_params:,}")

print(f"\nKey insights about color datasets:")
print(f"- RGB channels provide richer information than grayscale")
print(f"- First layer filters learn color-sensitive features")
print(f"- Batch normalization helps with training stability")
print(f"- Data augmentation improves generalization")
print(f"- More complex than MNIST, requires deeper networks")
