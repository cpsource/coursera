# Deep Neural Networks - Fixed and Improved Version

# Import the libraries we need for this lab
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
import time

torch.manual_seed(2)

print("🚀 Starting Deep Neural Networks Lab")
print("=" * 50)

# Create the model class using sigmoid as the activation function
class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x)) 
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x

# Create the model class using Tanh as a activation function
class NetTanh(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

# Create the model class using ReLU as a activation function
class NetRelu(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    
    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))  
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Train the model with improved progress tracking
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100, model_name="Model"):
    print(f"\n🏋️ Training {model_name}")
    print("-" * 30)
    
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    
    # Calculate total dataset size for accuracy calculation
    total_validation_samples = len(validation_loader.dataset)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set model to training mode
        epoch_loss = 0
        num_batches = 0
        
        print(f"Epoch {epoch+1}/{epochs} - Training...", end=" ")
        
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            useful_stuff['training_loss'].append(loss.item())
            
            # Print progress dots every few batches
            if (i + 1) % 5 == 0:
                print(".", end="")

        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct = 0
        total_val_loss = 0
        
        print(" Validating...", end=" ")
        
        with torch.no_grad():  # Disable gradient computation for validation
            for x, y in validation_loader:
                z = model(x.view(-1, 28 * 28))
                val_loss = criterion(z, y)
                total_val_loss += val_loss.item()
                
                _, label = torch.max(z, 1)
                correct += (label == y).sum().item()

        # Calculate metrics
        avg_epoch_loss = epoch_loss / num_batches
        accuracy = 100 * (correct / total_validation_samples)  # Fixed: use actual dataset size
        avg_val_loss = total_val_loss / len(validation_loader)
        
        useful_stuff['validation_accuracy'].append(accuracy)
        
        # Print epoch summary
        print(f" ✓")
        print(f"    Train Loss: {avg_epoch_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {accuracy:.2f}%")
    
    training_time = time.time() - start_time
    print(f"✅ {model_name} training completed in {training_time:.2f} seconds")
    print(f"Final validation accuracy: {useful_stuff['validation_accuracy'][-1]:.2f}%")
    
    return useful_stuff

# Create the datasets
print("\n📚 Loading MNIST dataset...")
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

# Create the criterion function
criterion = nn.CrossEntropyLoss()

# Create the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Set the parameters for creating the models
input_dim = 28 * 28
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10

print(f"\n🏗️ Model Architecture:")
print(f"Input dimension: {input_dim}")
print(f"Hidden layer 1: {hidden_dim1} neurons")
print(f"Hidden layer 2: {hidden_dim2} neurons") 
print(f"Output dimension: {output_dim}")

# Set the number of epochs
cust_epochs = 10
learning_rate = 0.01

print(f"\n⚙️ Training Parameters:")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {cust_epochs}")
print(f"Batch size: {train_loader.batch_size}")

# Train the model with sigmoid function
print(f"\n" + "="*50)
model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results_sigmoid = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs, model_name="Sigmoid Network")

# Train the model with tanh function
print(f"\n" + "="*50)
model_Tanh = NetTanh(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(model_Tanh.parameters(), lr=learning_rate)
training_results_tanh = train(model_Tanh, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs, model_name="Tanh Network")

# Train the model with ReLU function
print(f"\n" + "="*50)
modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs, model_name="ReLU Network")

print(f"\n" + "="*50)
print("📊 TRAINING COMPLETE - Generating Comparison Plots")

# Create subplots for better visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Compare the training loss
ax1.plot(training_results_tanh['training_loss'], label='Tanh', alpha=0.7)
ax1.plot(training_results_sigmoid['training_loss'], label='Sigmoid', alpha=0.7)
ax1.plot(training_results_relu['training_loss'], label='ReLU', alpha=0.7)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Batch Iterations')
ax1.set_title('Training Loss Over Iterations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Compare the validation accuracy
ax2.plot(training_results_tanh['validation_accuracy'], label='Tanh', marker='o', alpha=0.7)
ax2.plot(training_results_sigmoid['validation_accuracy'], label='Sigmoid', marker='s', alpha=0.7)
ax2.plot(training_results_relu['validation_accuracy'], label='ReLU', marker='^', alpha=0.7)
ax2.set_ylabel('Validation Accuracy (%)')
ax2.set_xlabel('Epoch')
ax2.set_title('Validation Accuracy Over Epochs')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final comparison
print("\n🏆 Final Results Summary:")
print("-" * 40)
final_sigmoid_acc = training_results_sigmoid['validation_accuracy'][-1]
final_tanh_acc = training_results_tanh['validation_accuracy'][-1]
final_relu_acc = training_results_relu['validation_accuracy'][-1]

print(f"Sigmoid Network: {final_sigmoid_acc:.2f}% accuracy")
print(f"Tanh Network:    {final_tanh_acc:.2f}% accuracy")
print(f"ReLU Network:    {final_relu_acc:.2f}% accuracy")

# Determine best performing model
best_model = max([
    ("Sigmoid", final_sigmoid_acc),
    ("Tanh", final_tanh_acc), 
    ("ReLU", final_relu_acc)
], key=lambda x: x[1])

print(f"\n🥇 Best performing activation function: {best_model[0]} ({best_model[1]:.2f}%)")
