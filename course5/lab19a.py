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
images = load_images_from_zip(zip_file_bytes)

print(f"Dataset Summary:")
print(f"- Anastasia images: {len(images['anastasia'])}")
print(f"- Takao images: {len(images['takao'])}")
print(f"- Total images: {len(images['anastasia']) + len(images['takao'])}")

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
