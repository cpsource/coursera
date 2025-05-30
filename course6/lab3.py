# Linear Classifier with PyTorch

from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim

#
# Download Data
#
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

import requests
import zipfile
import os
from pathlib import Path

def prepare_data(url, path="resources/data", overwrite=True):
    """
    Download and extract a zip file to a specified path.
    Similar to skillsnetwork.prepare() functionality.
    """
    # Create the directory if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists and handle overwrite
    if os.path.exists(path) and os.listdir(path) and not overwrite:
        print(f"Data already exists at {path} and overwrite=False")
        return
    
    # Download the file
    print("Downloading data...")
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for bad status codes
    
    # Save to a temporary zip file
    zip_path = os.path.join(path, "temp_download.zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    # Clean up the zip file
    os.remove(zip_path)
    print(f"Data prepared successfully at {path}")

if False:
    # Usage - equivalent to your original async call
    prepare_data(
        "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip",
        path="resources/data",
        overwrite=True
    )
    
# The path to all the images are stored in the variable directory.
directory="resources/data"

if False:
    #
    # Load negatives
    #

    negative='Negative'
    negative_file_path=os.path.join(directory,negative)
    print(negative_file_path)
    print(os.listdir(negative_file_path)[0:3])
    print([os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path)][0:3])
    negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
    negative_files.sort()
    print(negative_files[0:3])

    #
    # now load positives
    #

    positive = "Positive"
    positive_file_path = os.path.join(directory, positive)
    print(positive_file_path)
    print(os.listdir(positive_file_path)[0:3])
    print([os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path)][0:3])
    positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
    positive_files.sort()
    print(positive_files[0:3])

# some debug info
if False:
    # Open the first negative image
    image1 = Image.open(negative_files[0])
    # You can view the image directly
    # image1

    # Open the first positive image
    image2 = Image.open(positive_files[0])
    # You can view the image directly
    # image2
    # You can also display basic image information
    print(f"Negative image size: {image1.size}")
    print(f"Negative image mode: {image1.mode}")
    print(f"Positive image size: {image2.size}")
    print(f"Positive image mode: {image2.mode}")

    plt.imshow(image1)
    plt.title("1st Image With No Cracks")
    plt.show()

    image2 = Image.open(negative_files[1])
    plt.imshow(image2)
    plt.title("2nd Image With No Cracks")
    plt.show()

# Dataset class

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="resources/data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:10000] #Change to 30000 to use the full test dataset
            self.Y=self.Y[0:10000] #Change to 30000 to use the full test dataset
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)    
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)
        return image, y

# Transform Object and Dataset Object

# Define the normalization parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create the transform object using Compose
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert PIL Image to tensor
    transforms.Normalize(mean, std)   # Normalize with given mean and std
])

# Create dataset objects
dataset_train = Dataset(transform=transform, train=True)
dataset_val   = Dataset(transform=transform, train=False)

# Display dataset information
print(f"Training dataset size: {len(dataset_train)}")
print(f"Validation dataset size: {len(dataset_val)}")

# Example: Show a sample from the training dataset
if len(dataset_train) > 0:
    sample_image, sample_label = dataset_train[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample label: {sample_label}")
    print(f"Label meaning: {'Positive (crack)' if sample_label == 1 else 'Negative (no crack)'}")
   

# We can find the shape of the image:

print(dataset_train[0][0].shape)

# We see that it's a color image with three channels:

size_of_image=3*227*227
print(size_of_image)

# Custom Softmax Module for Two Classes
class SoftMax(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(SoftMax, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        # Flatten the image tensor to a 1D vector
        x = x.view(x.size(0), -1)  # Batch size x flattened features
        z = self.linear(x)
        return z

# Set random seed for reproducibility
torch.manual_seed(0)

# Create the model
input_size = size_of_image  # 3*227*227 = 154587
output_size = 2  # Two classes: crack vs no crack
model = SoftMax(input_size, output_size)

# Training parameters
learning_rate = 0.1
momentum = 0.1
batch_size = 5
epochs = 5

# Create data loaders
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training function
def train_model(model, train_loader, validation_loader, criterion, optimizer, epochs):
    # Lists to store training metrics
    training_loss = []
    validation_accuracy = []
    
    for epoch in range(epochs):
        print(f"\n=== EPOCH {epoch + 1}/{epochs} ===")
        # Training phase
        model.train()
        total_loss = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average training loss for this epoch
        avg_training_loss = total_loss / len(train_loader)
        training_loss.append(avg_training_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation accuracy
        val_accuracy = correct / total
        validation_accuracy.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_training_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    return training_loss, validation_accuracy

# Train the model
print("Starting training...")
print(f"Model input size: {input_size}")
print(f"Training dataset size: {len(dataset_train)}")
print(f"Validation dataset size: {len(dataset_val)}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Momentum: {momentum}")
print(f"Epochs: {epochs}")
print("-" * 50)

training_loss, validation_accuracy = train_model(model, train_loader, validation_loader, criterion, optimizer, epochs)

# Find and display the maximum validation accuracy
max_accuracy = max(validation_accuracy)
max_accuracy_epoch = validation_accuracy.index(max_accuracy) + 1

print("-" * 50)
print("Training completed!")
print(f"Validation accuracies for each epoch: {[f'{acc:.4f}' for acc in validation_accuracy]}")
print(f"Maximum validation accuracy: {max_accuracy:.4f} (achieved at epoch {max_accuracy_epoch})")

# Optional: Plot training progress
if True:  # Set to True if you want to see plots
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), training_loss, 'b-', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), validation_accuracy, 'r-', marker='o')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

