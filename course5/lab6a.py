# Neural Networks with One Hidden Layer

# Import the libraries we need for this lab

# Using the following line code to install the torchvision library
# !mamba install -y torchvision

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np

# Define a function to plot accuracy and loss

def plot_accuracy_loss(training_results):
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r')
    plt.ylabel('Loss')
    plt.title('Training Loss per Iteration')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(training_results['validation_accuracy'], 'b')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Define a function to plot model parameters

def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print(f"The following are the parameters for layer {count // 2 + 1}")
        if "bias" in ele:
            print(f"The size of bias: {model.state_dict()[ele].size()}")
        else:
            print(f"The size of weights: {model.state_dict()[ele].size()}")

# Define a function to display data

def show_data(data_sample, label=None, prediction=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(data_sample.numpy().reshape(28, 28), cmap='gray')
    if label is not None and prediction is not None:
        plt.title(f"True: {label}, Predicted: {prediction}")
    elif label is not None:
        plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# Define a Neural Network class

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

# Define a training function to train the model

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        
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
        
        # Store average loss for the epoch
        avg_loss = epoch_loss / num_batches
        useful_stuff['training_loss'].append(avg_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation for validation
            for x, y in validation_loader:
                z = model(x.view(-1, 28 * 28))
                _, predicted = torch.max(z.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        useful_stuff['validation_accuracy'].append(accuracy)
        
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Validation Accuracy = {accuracy:.2f}%")
    
    return useful_stuff

# Create training dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Create validating dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create criterion function
criterion = nn.CrossEntropyLoss()

# Create data loader for both train dataset and validate dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Create the model with 100 neurons
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model = Net(input_dim, hidden_dim, output_dim)

# Print the parameters for model
print("Model Architecture:")
print_model_parameters(model)
print()

# Set the learning rate and the optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
print("Training the custom Net model...")
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=100)

# Plot the accuracy and loss
print("\nTraining completed. Plotting results...")
plot_accuracy_loss(training_results)

# Plot the first five misclassified samples
print("Finding misclassified samples...")
model.eval()
count = 0
with torch.no_grad():
    for x, y in validation_dataset:
        z = model(x.reshape(-1, 28 * 28))
        _, yhat = torch.max(z, 1)
        if yhat != y:
            show_data(x, label=y, prediction=yhat.item())
            count += 1
        if count >= 5:
            break

# Practice: Use nn.Sequential to build the same model. Use plot_accuracy_loss to print out the accuracy and loss
print("\n" + "="*50)
print("Training with nn.Sequential model...")
print("="*50)

input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

model_sequential = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_dim, output_dim),
)

learning_rate = 0.01
optimizer_sequential = torch.optim.SGD(model_sequential.parameters(), lr=learning_rate)

training_results_sequential = train(model_sequential, criterion, train_loader, validation_loader, 
                                  optimizer_sequential, epochs=10)

print("\nSequential model training completed. Plotting results...")
plot_accuracy_loss(training_results_sequential)
