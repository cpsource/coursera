# Test Sigmoid, Tanh, and ReLU Activation Functions on the MNIST Dataset

# Uncomment the following line to install the torchvision library
# !mamba install -y torchvision

# Import the libraries we need for this lab
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np

# Build the model with sigmoid function
class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

# Build the model with Tanh function
class NetTanh(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x

# Build the model with ReLU function
class NetRelu(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Define the function for training the model
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100, model_name="Model"):
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    
    print(f"\nStarting training for {model_name}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            useful_stuff['training_loss'].append(loss.item())
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation for efficiency
            for x, y in validation_loader:
                z = model(x.view(-1, 28 * 28))
                _, predicted = torch.max(z, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * (correct / total)
        avg_loss = epoch_loss / num_batches
        useful_stuff['validation_accuracy'].append(accuracy)
        
        # Print progress every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    
    print(f"Training completed for {model_name}")
    print(f"Final validation accuracy: {accuracy:.2f}%")
    print("-" * 50)
    
    return useful_stuff

# Create the datasets
print("Loading MNIST dataset...")

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

# Create the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Create the criterion function
criterion = nn.CrossEntropyLoss()

# Set model parameters
input_dim = 28 * 28  # 784 pixels
hidden_dim = 100
output_dim = 10  # 10 digit classes (0-9)
learning_rate = 0.01
epochs = 30

print(f"\nModel Configuration:")
print(f"Input dimension: {input_dim}")
print(f"Hidden dimension: {hidden_dim}")
print(f"Output dimension: {output_dim}")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {epochs}")

# Train model with sigmoid activation
print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

model_sigmoid = Net(input_dim, hidden_dim, output_dim)
optimizer_sigmoid = torch.optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
training_results_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, 
                                optimizer_sigmoid, epochs=epochs, model_name="Sigmoid")

# Train model with Tanh activation
model_tanh = NetTanh(input_dim, hidden_dim, output_dim)
optimizer_tanh = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
training_results_tanh = train(model_tanh, criterion, train_loader, validation_loader, 
                             optimizer_tanh, epochs=epochs, model_name="Tanh")

# Train model with ReLU activation
model_relu = NetRelu(input_dim, hidden_dim, output_dim)
optimizer_relu = torch.optim.SGD(model_relu.parameters(), lr=learning_rate)
training_results_relu = train(model_relu, criterion, train_loader, validation_loader, 
                             optimizer_relu, epochs=epochs, model_name="ReLU")

# Compare the training loss
print("\nGenerating comparison plots...")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_results_tanh['training_loss'], label='Tanh', alpha=0.7)
plt.plot(training_results_sigmoid['training_loss'], label='Sigmoid', alpha=0.7)
plt.plot(training_results_relu['training_loss'], label='ReLU', alpha=0.7)
plt.ylabel('Loss')
plt.xlabel('Training Iterations')
plt.title('Training Loss Comparison Across Activation Functions')
plt.legend()
plt.grid(True, alpha=0.3)

# Compare the validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_results_tanh['validation_accuracy'], label='Tanh', marker='o', markersize=3)
plt.plot(training_results_sigmoid['validation_accuracy'], label='Sigmoid', marker='s', markersize=3)
plt.plot(training_results_relu['validation_accuracy'], label='ReLU', marker='^', markersize=3)
plt.ylabel('Validation Accuracy (%)')
plt.xlabel('Epochs')
plt.title('Validation Accuracy Comparison Across Activation Functions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final results summary
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"Final Validation Accuracies:")
print(f"  Sigmoid: {training_results_sigmoid['validation_accuracy'][-1]:.2f}%")
print(f"  Tanh:    {training_results_tanh['validation_accuracy'][-1]:.2f}%")
print(f"  ReLU:    {training_results_relu['validation_accuracy'][-1]:.2f}%")

best_activation = max([
    ('Sigmoid', training_results_sigmoid['validation_accuracy'][-1]),
    ('Tanh', training_results_tanh['validation_accuracy'][-1]),
    ('ReLU', training_results_relu['validation_accuracy'][-1])
], key=lambda x: x[1])

print(f"\nBest performing activation function: {best_activation[0]} ({best_activation[1]:.2f}%)")
