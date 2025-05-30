# Convolutional Neural Network for Anime Image Classification - FIXED VERSION

# Importing Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from PIL import Image
import io
import requests
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

print("=" * 60)
print("ANIME CHARACTER CLASSIFICATION WITH CNN")
print("=" * 60)

# Load dataset
def load_images_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        images = {'anastasia': [], 'takao': []}
        print(f"Loading images from zip file...")
        
        for file_name in zip_ref.namelist():
            if file_name.startswith('anastasia') and file_name.endswith('.jpg'):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert('RGB')
                    images['anastasia'].append(np.array(img))
            elif file_name.startswith('takao') and file_name.endswith('.jpg'):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert('RGB')
                    images['takao'].append(np.array(img))
        
        print(f"✓ Loaded {len(images['anastasia'])} Anastasia images")
        print(f"✓ Loaded {len(images['takao'])} Takao images")
    return images

zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/xZQHOyN8ONT92kH-ASb4Pw/data.zip'

print("Downloading dataset...")
# Download the ZIP file
response = requests.get(zip_file_url)
zip_file_bytes = io.BytesIO(response.content)

# Load images from zip file
original_images = load_images_from_zip(zip_file_bytes)

print(f"Dataset Summary:")
print(f"- Anastasia images: {len(original_images['anastasia'])}")
print(f"- Takao images: {len(original_images['takao'])}")
print(f"- Total images: {len(original_images['anastasia']) + len(original_images['takao'])}")

# Keep reference for later use
images = original_images

# Define Custom Dataset Class
class AnimeDataset(Dataset):
    def __init__(self, images, transform=None, classes=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.classes = classes

        print(f"Creating dataset with classes: {classes}")
        for label, class_name in enumerate(self.classes):
            for img in images[class_name]:
                self.images.append(img)
                self.labels.append(label)
        
        print(f"✓ Dataset created with {len(self.images)} total samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
print("\nSetting up data transformations...")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
print("✓ Transforms configured: Resize(64x64), ToTensor, Normalize")

# Load dataset
dataset = AnimeDataset(images, transform=transform, classes=['anastasia', 'takao'])

# Split Dataset into Training and Validation Sets
print("\nSplitting dataset...")
# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Generate a list of indices for the entire dataset
indices = list(range(len(dataset)))

# Split the indices into training and validation sets
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=seed)

# Create samplers for training and validation sets
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create DataLoader objects for training and validation sets
train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=20, sampler=val_sampler)

print(f"✓ Training set size: {len(train_indices)} samples")
print(f"✓ Validation set size: {len(val_indices)} samples")
print(f"✓ Train/Val split: {len(train_indices)/(len(train_indices)+len(val_indices))*100:.1f}%/{len(val_indices)/(len(train_indices)+len(val_indices))*100:.1f}%")

# Define the CNN Model
print("\nDefining CNN Architecture...")

class AnimeCNN(nn.Module):
    def __init__(self):
        super(AnimeCNN, self).__init__()
        # Add padding=1 to maintain the border
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = AnimeCNN()
print("✓ CNN Model created")
print(f"Model architecture:\n{model}")

# Print tensor shapes through the network
print("\nAnalyzing network dimensions...")
input_tensor = torch.randn(1, 3, 64, 64)

def print_size(module, input, output):
    print(f"  {module.__class__.__name__}: {list(input[0].shape)} → {list(output.shape)}")

# Register hooks
hooks = []
for layer in model.children():
    hook = layer.register_forward_hook(print_size)
    hooks.append(hook)

# Inspect output sizes
with torch.no_grad():
    output = model(input_tensor)
    
print(f"✓ Final output shape: {list(output.shape)} (batch_size=1, num_classes=2)")

# Remove hooks
for hook in hooks:
    hook.remove()

# Define Loss Function and Optimizer
print("\nSetting up training components...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("✓ Loss function: CrossEntropyLoss")
print("✓ Optimizer: Adam (lr=0.001)")

# Train the Model
print("\n" + "="*50)
print("STARTING TRAINING - ORIGINAL MODEL")
print("="*50)

num_epochs = 5
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val
    val_losses.append(val_loss)

    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

print('\n✓ Original model training completed!')

# Visualize training progress
print("\nPlotting original model training progress...")

# Create figure with subplots
fig = plt.figure(figsize=(18, 6))

# Training loss plot
plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(val_losses, 'r--', label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Original Model: Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample predictions - use a separate figure approach
plt.subplot(1, 3, 2)
model.eval()
data_iter = iter(val_loader)
sample_images, sample_labels = next(data_iter)
outputs = model(sample_images)
_, predicted = torch.max(outputs, 1)

# Show just one sample image as a preview
sample_img = sample_images[0] / 2 + 0.5  # unnormalize
npimg = sample_img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.axis('off')

actual = sample_labels[0].item()
pred = predicted[0].item()
color = 'green' if actual == pred else 'red'
classes = ['Anastasia', 'Takao']
plt.title(f'Sample Prediction\nTrue: {classes[actual]} | Pred: {classes[pred]}', color=color)

# Accuracy summary
plt.subplot(1, 3, 3)
correct_batch = (predicted == sample_labels).sum().item()
total_batch = sample_labels.size(0)
batch_accuracy = 100 * correct_batch / total_batch

# Create accuracy visualization
classes = ['Anastasia', 'Takao']
plt.bar(classes, [100, 100], color='lightgray', alpha=0.3, label='Total')
plt.bar(classes, [batch_accuracy, batch_accuracy], color='blue', alpha=0.7, label='Correct')
plt.ylabel('Accuracy (%)')
plt.title(f'Batch Accuracy: {batch_accuracy:.1f}%')
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()
plt.show()

# Show detailed predictions in a separate figure
print("Showing detailed sample predictions...")
fig2, axes = plt.subplots(2, 4, figsize=(12, 6))
fig2.suptitle('Original Model: Detailed Sample Predictions', fontsize=14)
axes = axes.flatten()

def imshow_prediction(img, ax, actual, predicted, classes=['Anastasia', 'Takao']):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    color = 'green' if actual == predicted else 'red'
    ax.set_title(f'True: {classes[actual]}\nPred: {classes[predicted]}', 
                color=color, fontsize=9)
    ax.axis('off')

for idx in range(min(8, len(sample_images))):
    imshow_prediction(sample_images[idx].cpu(), axes[idx], 
                     sample_labels[idx].item(), predicted[idx].item())

# Hide unused subplots
for idx in range(len(sample_images), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Calculate final accuracy
correct = 0
total = 0
class_correct = [0, 0]
class_total = [0, 0]

model.eval()
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == labels[i]).item()
            class_total[label] += 1

original_accuracy = 100 * correct / total
print(f'\nOriginal Model Final Validation Accuracy: {original_accuracy:.2f}%')
print(f'Anastasia Classification Accuracy: {100 * class_correct[0] / class_total[0]:.2f}%')
print(f'Takao Classification Accuracy: {100 * class_correct[1] / class_total[1]:.2f}%')

# EXERCISE 1 - Modified CNN with LeakyReLU
print("\n" + "="*50)
print("EXERCISE 1: LEAKY RELU ACTIVATION")
print("="*50)

class AnimeCNNModified(nn.Module):
    def __init__(self):
        super(AnimeCNNModified, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))  # Changed to LeakyReLU
        x = self.pool(F.leaky_relu(self.conv2(x)))  # Changed to LeakyReLU
        x = x.view(-1, 64 * 16 * 16)
        x = F.leaky_relu(self.fc1(x))              # Changed to LeakyReLU
        x = self.fc2(x)
        return x

model_leaky = AnimeCNNModified()
criterion_leaky = nn.CrossEntropyLoss()
optimizer_leaky = optim.Adam(model_leaky.parameters(), lr=0.001)

print("✓ Modified model created with LeakyReLU activation")
print("LeakyReLU allows small negative values (vs ReLU which zeros them)")

# Train LeakyReLU model
num_epochs = 5
train_losses_leaky = []
val_losses_leaky = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Training
    model_leaky.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        optimizer_leaky.zero_grad()
        outputs = model_leaky(inputs)
        loss = criterion_leaky(outputs, labels)
        loss.backward()
        optimizer_leaky.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses_leaky.append(train_loss)

    # Validation
    model_leaky.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model_leaky(inputs)
            loss = criterion_leaky(outputs, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    val_losses_leaky.append(val_loss)
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# Calculate LeakyReLU model accuracy
correct_leaky = 0
total_leaky = 0
model_leaky.eval()
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model_leaky(images)
        _, predicted = torch.max(outputs.data, 1)
        total_leaky += labels.size(0)
        correct_leaky += (predicted == labels).sum().item()

leaky_accuracy = 100 * correct_leaky / total_leaky
print(f'\n✓ LeakyReLU Model Accuracy: {leaky_accuracy:.2f}%')

# Plot LeakyReLU training progress
print("\nPlotting LeakyReLU model comparison...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', label='Original ReLU Train', linewidth=2)
plt.plot(val_losses, 'b--', label='Original ReLU Val', linewidth=2)
plt.plot(train_losses_leaky, 'g-', label='LeakyReLU Train', linewidth=2)
plt.plot(val_losses_leaky, 'g--', label='LeakyReLU Val', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ReLU vs LeakyReLU Training Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
models = ['ReLU', 'LeakyReLU']
accuracies = [original_accuracy, leaky_accuracy]
colors = ['blue', 'green']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy (%)')
plt.title('Activation Function Comparison')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', fontweight='bold')

plt.subplot(1, 3, 3)
# Show LeakyReLU sample predictions
model_leaky.eval()
data_iter = iter(val_loader)
images, labels = next(data_iter)
outputs = model_leaky(images)
_, predicted = torch.max(outputs, 1)

# Create a mini subplot for predictions
plt.axis('off')
plt.title('LeakyReLU Sample Predictions')
plt.text(0.5, 0.5, f'Sample Results:\nCorrect: {(predicted[:8] == labels[:8]).sum().item()}/8', 
         ha='center', va='center', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

plt.tight_layout()
plt.show()

# EXERCISE 2 - Increased Epochs
print("\n" + "="*50)
print("EXERCISE 2: INCREASED EPOCHS (7 vs 5)")
print("="*50)

model_extended = AnimeCNN()
criterion_extended = nn.CrossEntropyLoss()
optimizer_extended = optim.Adam(model_extended.parameters(), lr=0.001)

num_epochs_extended = 7
train_losses_extended = []
val_losses_extended = []

print(f"Training for {num_epochs_extended} epochs (vs original {num_epochs})")

for epoch in range(num_epochs_extended):
    print(f"\nEpoch {epoch+1}/{num_epochs_extended}")
    
    # Training
    model_extended.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        optimizer_extended.zero_grad()
        outputs = model_extended(inputs)
        loss = criterion_extended(outputs, labels)
        loss.backward()
        optimizer_extended.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses_extended.append(train_loss)

    # Validation
    model_extended.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model_extended(inputs)
            loss = criterion_extended(outputs, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    val_losses_extended.append(val_loss)
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# Calculate extended training accuracy
correct_extended = 0
total_extended = 0
model_extended.eval()
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model_extended(images)
        _, predicted = torch.max(outputs.data, 1)
        total_extended += labels.size(0)
        correct_extended += (predicted == labels).sum().item()

extended_accuracy = 100 * correct_extended / total_extended
print(f'\n✓ Extended Training Model Accuracy: {extended_accuracy:.2f}%')

# Plot extended training comparison
print("\nPlotting extended training comparison...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, 6), train_losses, 'b-', label='5 Epochs Train', linewidth=2, marker='o')
plt.plot(range(1, 6), val_losses, 'b--', label='5 Epochs Val', linewidth=2, marker='o')
plt.plot(range(1, 8), train_losses_extended, 'r-', label='7 Epochs Train', linewidth=2, marker='s')
plt.plot(range(1, 8), val_losses_extended, 'r--', label='7 Epochs Val', linewidth=2, marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Duration Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
training_types = ['5 Epochs', '7 Epochs']
accuracies = [original_accuracy, extended_accuracy]
colors = ['blue', 'red']
bars = plt.bar(training_types, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy (%)')
plt.title('Training Duration Impact')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', fontweight='bold')

plt.subplot(1, 3, 3)
# Check for overfitting signs
epochs_5 = range(1, 6)
epochs_7 = range(1, 8)
plt.plot(epochs_5, np.array(train_losses) - np.array(val_losses), 'b-', 
         label='5 Epochs Gap', linewidth=2, marker='o')
plt.plot(epochs_7, np.array(train_losses_extended) - np.array(val_losses_extended), 'r-', 
         label='7 Epochs Gap', linewidth=2, marker='s')
plt.xlabel('Epochs')
plt.ylabel('Train Loss - Val Loss')
plt.title('Overfitting Detection\n(Lower = Better)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# EXERCISE 3 - Different Character Classes
print("\n" + "="*50)
print("EXERCISE 3: NEW CHARACTER CLASSES")
print("="*50)

def load_new_images_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        images = {'arcueid_brunestud': [], 'yukinoshita_yukino': []}
        print("Loading new character dataset...")
        
        for file_name in zip_ref.namelist():
            if file_name.startswith('arcueid_brunestud') and file_name.endswith('.jpg'):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert('RGB')
                    images['arcueid_brunestud'].append(np.array(img))
            elif file_name.startswith('yukinoshita_yukino') and file_name.endswith('.jpg'):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert('RGB')
                    images['yukinoshita_yukino'].append(np.array(img))
        
        print(f"✓ Loaded {len(images['arcueid_brunestud'])} Arcueid images")
        print(f"✓ Loaded {len(images['yukinoshita_yukino'])} Yukino images")
    return images

# Load new dataset
new_zip_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/yNB99OssvDWOrNTHf2Yuxw/data-practice.zip'
print("Downloading new character dataset...")

response_new = requests.get(new_zip_url)
zip_file_bytes_new = io.BytesIO(response_new.content)
new_images = load_new_images_from_zip(zip_file_bytes_new)

print(f"New Dataset Summary:")
print(f"- Arcueid images: {len(new_images['arcueid_brunestud'])}")
print(f"- Yukino images: {len(new_images['yukinoshita_yukino'])}")

# Create new dataset and dataloaders
new_dataset = AnimeDataset(new_images, transform=transform, 
                          classes=['arcueid_brunestud', 'yukinoshita_yukino'])

# Split new dataset
new_indices = list(range(len(new_dataset)))
new_train_indices, new_val_indices = train_test_split(new_indices, test_size=0.2, random_state=seed)

new_train_sampler = SubsetRandomSampler(new_train_indices)
new_val_sampler = SubsetRandomSampler(new_val_indices)

new_train_loader = DataLoader(new_dataset, batch_size=8, sampler=new_train_sampler)
new_val_loader = DataLoader(new_dataset, batch_size=20, sampler=new_val_sampler)

print(f"✓ New training set: {len(new_train_indices)} samples")
print(f"✓ New validation set: {len(new_val_indices)} samples")

# Train model on new dataset
model_new_chars = AnimeCNN()
criterion_new = nn.CrossEntropyLoss()
optimizer_new = optim.Adam(model_new_chars.parameters(), lr=0.001)

num_epochs_new = 5
train_losses_new = []
val_losses_new = []

print(f"\nTraining on new character classes...")

for epoch in range(num_epochs_new):
    print(f"\nEpoch {epoch+1}/{num_epochs_new}")
    
    # Training
    model_new_chars.train()
    running_loss = 0.0
    for data in new_train_loader:
        inputs, labels = data
        optimizer_new.zero_grad()
        outputs = model_new_chars(inputs)
        loss = criterion_new(outputs, labels)
        loss.backward()
        optimizer_new.step()
        running_loss += loss.item()

    train_loss = running_loss / len(new_train_loader)
    train_losses_new.append(train_loss)

    # Validation
    model_new_chars.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in new_val_loader:
            inputs, labels = data
            outputs = model_new_chars(inputs)
            loss = criterion_new(outputs, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(new_val_loader)
    val_losses_new.append(val_loss)
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# Calculate new characters accuracy
correct_new = 0
total_new = 0
model_new_chars.eval()
with torch.no_grad():
    for data in new_val_loader:
        images, labels = data
        outputs = model_new_chars(images)
        _, predicted = torch.max(outputs.data, 1)
        total_new += labels.size(0)
        correct_new += (predicted == labels).sum().item()

new_chars_accuracy = 100 * correct_new / total_new
print(f'\n✓ New Characters Model Accuracy: {new_chars_accuracy:.2f}%')

# Plot new characters dataset comparison
print("\nPlotting new characters dataset analysis...")
plt.figure(figsize=(15, 10))

# Dataset comparison
plt.subplot(2, 3, 1)
datasets = ['Anastasia\nvs\nTakao', 'Arcueid\nvs\nYukino']
accuracies = [original_accuracy, new_chars_accuracy]
colors = ['blue', 'purple']
bars = plt.bar(datasets, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy (%)')
plt.title('Character Pair Comparison')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', fontweight='bold')

# Training curves comparison
plt.subplot(2, 3, 2)
plt.plot(train_losses, 'b-', label='Original Train', linewidth=2)
plt.plot(val_losses, 'b--', label='Original Val', linewidth=2)
plt.plot(train_losses_new, 'purple', label='New Chars Train', linewidth=2, linestyle='-')
plt.plot(val_losses_new, 'purple', label='New Chars Val', linewidth=2, linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Sample images from original dataset
plt.subplot(2, 3, 3)
plt.axis('off')
plt.title('Original Dataset Sample')
try:
    data_iter = iter(val_loader)
    images_sample, labels_sample = next(data_iter)
    sample_img = images_sample[0]
    sample_img = sample_img / 2 + 0.5  # unnormalize
    npimg = sample_img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
except Exception as e:
    plt.text(0.5, 0.5, 'Sample image\nnot available', ha='center', va='center')

# Sample images from new dataset  
plt.subplot(2, 3, 4)
plt.axis('off')
plt.title('New Dataset Sample')
try:
    data_iter_new = iter(new_val_loader)
    images_new, labels_new = next(data_iter_new)
    sample_img_new = images_new[0]
    sample_img_new = sample_img_new / 2 + 0.5  # unnormalize
    npimg_new = sample_img_new.numpy()
    plt.imshow(np.transpose(npimg_new, (1, 2, 0)))
except Exception as e:
    plt.text(0.5, 0.5, 'Sample image\nnot available', ha='center', va='center')

# All models comparison
plt.subplot(2, 3, 5)
all_models = ['Original\n(ReLU)', 'LeakyReLU', 'Extended\nTraining', 'New\nCharacters']
all_accuracies = [original_accuracy, leaky_accuracy, extended_accuracy, new_chars_accuracy]
all_colors = ['blue', 'green', 'red', 'purple']
bars = plt.bar(all_models, all_accuracies, color=all_colors, alpha=0.7)
plt.ylabel('Accuracy (%)')
plt.title('All Models Comparison')
plt.ylim(0, 100)
plt.xticks(rotation=45)
for bar, acc in zip(bars, all_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=9)

# Dataset statistics
plt.subplot(2, 3, 6)
plt.axis('off')
plt.title('Dataset Statistics')
stats_text = f"""Original Dataset:
• Anastasia: {len(original_images['anastasia'])} images
• Takao: {len(original_images['takao'])} images
• Total: {len(original_images['anastasia']) + len(original_images['takao'])}

New Dataset:
• Arcueid: {len(new_images['arcueid_brunestud'])} images  
• Yukino: {len(new_images['yukinoshita_yukino'])} images
• Total: {len(new_images['arcueid_brunestud']) + len(new_images['yukinoshita_yukino'])}

Train/Val Split: 80%/20%"""

plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.show()

# FINAL CONCLUSIONS AND RESULTS
print("\n" + "="*60)
print("FINAL RESULTS AND CONCLUSIONS")
print("="*60)

print(f"\n📊 ACCURACY COMPARISON:")
print(f"{'Model':<25} {'Accuracy':<10} {'Improvement'}")
print("-" * 45)
print(f"{'Original (ReLU)':<25} {original_accuracy:<10.2f}% {'(baseline)'}")
print(f"{'LeakyReLU':<25} {leaky_accuracy:<10.2f}% {leaky_accuracy-original_accuracy:+.2f}%")
print(f"{'Extended Training':<25} {extended_accuracy:<10.2f}% {extended_accuracy-original_accuracy:+.2f}%")
print(f"{'New Characters':<25} {new_chars_accuracy:<10.2f}% {'(different task)'}")

print(f"\n🔍 KEY FINDINGS:")

# Activation function analysis
if leaky_accuracy > original_accuracy:
    print(f"✅ LeakyReLU improved accuracy by {leaky_accuracy-original_accuracy:.2f}%")
    print("   LeakyReLU allows small negative gradients, helping with vanishing gradient problem")
elif leaky_accuracy < original_accuracy:
    print(f"⚠️  LeakyReLU decreased accuracy by {original_accuracy-leaky_accuracy:.2f}%")
    print("   For this simple network, ReLU might be sufficient")
else:
    print("🔄 LeakyReLU showed similar performance to ReLU")

# Training duration analysis
if extended_accuracy > original_accuracy:
    print(f"✅ Extended training improved accuracy by {extended_accuracy-original_accuracy:.2f}%")
    print("   More epochs allowed better convergence")
elif extended_accuracy < original_accuracy:
    print(f"⚠️  Extended training showed signs of overfitting ({original_accuracy-extended_accuracy:.2f}% decrease)")
else:
    print("🔄 Extended training showed similar final performance")

# Dataset complexity analysis
print(f"\n📈 DATASET INSIGHTS:")
print(f"• Original dataset (Anastasia vs Takao): {original_accuracy:.1f}% accuracy")
print(f"• New dataset (Arcueid vs Yukino): {new_chars_accuracy:.1f}% accuracy")

if abs(new_chars_accuracy - original_accuracy) < 5:
    print("• Both character pairs showed similar classification difficulty")
elif new_chars_accuracy > original_accuracy:
    print("• New character pair was easier to distinguish")
else:
    print("• New character pair was more challenging to distinguish")

print(f"\n🎯 RECOMMENDATIONS:")
best_accuracy = max(original_accuracy, leaky_accuracy, extended_accuracy)
if best_accuracy == leaky_accuracy:
    print("• Use LeakyReLU activation for best performance")
elif best_accuracy == extended_accuracy:
    print("• Use extended training (7+ epochs) for best performance")
else:
    print("• Original configuration (ReLU, 5 epochs) is optimal")

print(f"• Consider data augmentation to improve generalization")
print(f"• Monitor training/validation loss curves to detect overfitting")
print(f"• Current model achieves {best_accuracy:.1f}% accuracy - consider deeper networks for improvement")

print(f"\n✅ Analysis complete! Best performing model achieved {best_accuracy:.2f}% accuracy.")

# Final comprehensive visualization
print("\nGenerating final comprehensive analysis plots...")
plt.figure(figsize=(20, 12))

# 1. All training curves together
plt.subplot(3, 4, 1)
plt.plot(train_losses, 'b-', label='Original Train', linewidth=2)
plt.plot(val_losses, 'b--', label='Original Val', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Original Model Training')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 2)
plt.plot(train_losses_leaky, 'g-', label='LeakyReLU Train', linewidth=2)
plt.plot(val_losses_leaky, 'g--', label='LeakyReLU Val', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('LeakyReLU Model Training')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 3)
plt.plot(train_losses_extended, 'r-', label='Extended Train', linewidth=2)
plt.plot(val_losses_extended, 'r--', label='Extended Val', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Extended Training (7 epochs)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 4, 4)
plt.plot(train_losses_new, 'purple', label='New Chars Train', linewidth=2, linestyle='-')
plt.plot(val_losses_new, 'purple', label='New Chars Val', linewidth=2, linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('New Characters Training')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Accuracy comparison bar chart
plt.subplot(3, 4, 5)
models = ['Original', 'LeakyReLU', 'Extended', 'New Chars']
accuracies = [original_accuracy, leaky_accuracy, extended_accuracy, new_chars_accuracy]
colors = ['blue', 'green', 'red', 'purple']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.ylabel('Accuracy (%)')
plt.title('Model Performance Comparison')
plt.ylim(0, 100)
plt.xticks(rotation=45)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', fontweight='bold')

# 3. Improvement over baseline
plt.subplot(3, 4, 6)
improvements = [0, leaky_accuracy-original_accuracy, 
               extended_accuracy-original_accuracy, new_chars_accuracy-original_accuracy]
improvement_colors = ['gray', 'green' if improvements[1]>0 else 'red',
                     'green' if improvements[2]>0 else 'red', 'purple']
bars = plt.bar(models, improvements, color=improvement_colors, alpha=0.7)
plt.ylabel('Accuracy Change (%)')
plt.title('Improvement Over Baseline')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xticks(rotation=45)
for bar, imp in zip(bars, improvements):
    if imp != 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.5 if imp>0 else -0.5), 
                 f'{imp:+.1f}%', ha='center', fontweight='bold')

# 4. Loss convergence comparison
plt.subplot(3, 4, 7)
final_train_losses = [train_losses[-1], train_losses_leaky[-1], 
                     train_losses_extended[-1], train_losses_new[-1]]
final_val_losses = [val_losses[-1], val_losses_leaky[-1], 
                   val_losses_extended[-1], val_losses_new[-1]]

x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, final_train_losses, width, label='Train Loss', alpha=0.7)
plt.bar(x + width/2, final_val_losses, width, label='Val Loss', alpha=0.7)
plt.ylabel('Final Loss')
plt.title('Final Loss Comparison')
plt.xticks(x, models, rotation=45)
plt.legend()

# 5-8. Sample predictions for each model
def show_predictions(model, loader, subplot_pos, title, class_names=['Class 0', 'Class 1']):
    plt.subplot(3, 4, subplot_pos)
    try:
        model.eval()
        data_iter = iter(loader)
        sample_images, sample_labels = next(data_iter)
        with torch.no_grad():
            outputs = model(sample_images)
            _, predicted = torch.max(outputs, 1)
        
        # Show accuracy for this batch
        correct = (predicted == sample_labels).sum().item()
        total = sample_labels.size(0)
        batch_acc = 100 * correct / total
        
        plt.axis('off')
        plt.title(f'{title}\nBatch Accuracy: {batch_acc:.1f}%')
        
        # Show one sample image
        sample_img = sample_images[0] / 2 + 0.5  # unnormalize
        npimg = sample_img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
        # Add prediction text
        pred_text = f"True: {class_names[sample_labels[0]]}\nPred: {class_names[predicted[0]]}"
        color = 'green' if sample_labels[0] == predicted[0] else 'red'
        plt.text(32, -5, pred_text, ha='center', fontweight='bold', color=color)
        
    except Exception as e:
        plt.axis('off')
        plt.title(f'{title}\nSample not available')
        plt.text(0.5, 0.5, 'Prediction\nsample\nunavailable', ha='center', va='center')

show_predictions(model, val_loader, 8, 'Original Model', ['Anastasia', 'Takao'])
show_predictions(model_leaky, val_loader, 9, 'LeakyReLU Model', ['Anastasia', 'Takao'])
show_predictions(model_extended, val_loader, 10, 'Extended Model', ['Anastasia', 'Takao'])
show_predictions(model_new_chars, new_val_loader, 11, 'New Characters', ['Arcueid', 'Yukino'])

# 12. Summary statistics
plt.subplot(3, 4, 12)
plt.axis('off')
plt.title('Experiment Summary')

# Find best model
best_idx = np.argmax(accuracies[:3])  # Exclude new chars as it's different task
best_model_name = models[best_idx]

summary_text = f"""
TARGET BEST MODEL: {best_model_name}
   Accuracy: {max(accuracies[:3]):.2f}%

RESULTS SUMMARY:
• Original (ReLU): {original_accuracy:.1f}%
• LeakyReLU: {leaky_accuracy:.1f}%  
• Extended Training: {extended_accuracy:.1f}%
• New Characters: {new_chars_accuracy:.1f}%

KEY INSIGHTS:
• Activation function impact: {leaky_accuracy-original_accuracy:+.1f}%
• Training duration impact: {extended_accuracy-original_accuracy:+.1f}%
• Dataset complexity varies by character pair

RECOMMENDATIONS:
• Monitor overfitting with val loss
• Consider data augmentation
• Try deeper architectures for improvement
"""

plt.text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

plt.suptitle('Comprehensive CNN Analysis: Anime Character Classification', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("COMPLETE ANALYSIS FINISHED!")
print("="*60)
print("All plots generated successfully. Check the visualizations above for detailed insights!")
