# PyTorch Pre-Trained Models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import requests
import zipfile
import os
from pathlib import Path

# Download data function (same as before)
use_directory = "concrete_data_week3"

def prepare_data(url, path=use_directory, overwrite=True):
    """
    Download and extract a zip file to a specified path.
    Similar to skillsnetwork.prepare() functionality.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(path) and os.listdir(path) and not overwrite:
        print(f"Data already exists at {path} and overwrite=False")
        return
    
    print("Downloading data...")
    response = requests.get(url)
    response.raise_for_status()
    
    zip_path = os.path.join(path, "temp_download.zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    os.remove(zip_path)
    print(f"Data prepared successfully at {path}")

# Uncomment to download data
# prepare_data(
#     "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip",
#     path=use_directory,
#     overwrite=True
# )

# Global constants
num_classes = 2
image_resize = 224
batch_size_training = 100
batch_size_validation = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations (equivalent to ImageDataGenerator)
train_transform = transforms.Compose([
    transforms.Resize((image_resize, image_resize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((image_resize, image_resize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Create datasets (equivalent to flow_from_directory)
print("... creating train_dataset")
train_dataset = datasets.ImageFolder(
    root=os.path.join(use_directory, "concrete_data_week3", "train"),
    transform=train_transform
)

print("... creating validation_dataset")
val_dataset = datasets.ImageFolder(
    root=os.path.join(use_directory, "concrete_data_week3", "valid"),
    transform=val_transform
)

# Create data loaders (equivalent to generators)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size_training, 
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size_validation, 
    shuffle=False,
    num_workers=2
)

print("...complete")
print(f"Classes found: {train_dataset.classes}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Build Model (equivalent to Sequential with ResNet50)
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Freeze ResNet layers (equivalent to trainable=False)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        # ResNet50's fc layer has 2048 input features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Create model instance
model = ResNetClassifier(num_classes=num_classes)
model = model.to(device)

# Print model structure (equivalent to model.summary())
print("\nModel structure:")
print(model)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Define loss function and optimizer (equivalent to compile)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 2

# Training function (equivalent to fit_generator/fit)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history

# Start training (equivalent to model.fit)
print("Starting training...")
fit_history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Save model (equivalent to model.save)
torch.save(model.state_dict(), 'classifier_resnet_model.pth')
torch.save(model, 'classifier_resnet_model_complete.pth')  # Save complete model
print("\nModel saved successfully!")

# Print final results
print("\nTraining completed!")
print("Final results:")
print(f"Final training accuracy: {fit_history['train_acc'][-1]:.4f}")
print(f"Final validation accuracy: {fit_history['val_acc'][-1]:.4f}")

