# Test Uniform, Default and He Initialization on MNIST Dataset with ReLU Activation
# Fixed version with progress tracking and bug fixes

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

torch.manual_seed(0)
print("PyTorch version:", torch.__version__)
print("Random seed set to 0 for reproducibility\n")

# Define the class for neural network model with He Initialization
class Net_He(nn.Module):
    def __init__(self, Layers):
        super(Net_He, self).__init__()
        self.hidden = nn.ModuleList()
        print(f"Creating He-initialized network with layers: {Layers}")

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)
            print(f"  Layer: {input_size} -> {output_size} (He initialization)")

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = F.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define the class for neural network model with Uniform Initialization
class Net_Uniform(nn.Module):
    def __init__(self, Layers):
        super(Net_Uniform, self).__init__()
        self.hidden = nn.ModuleList()
        print(f"Creating Uniform-initialized network with layers: {Layers}")

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)
            print(f"  Layer: {input_size} -> {output_size} (Uniform initialization)")

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = F.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define the class for neural network model with PyTorch Default Initialization
class Net(nn.Module):
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        print(f"Creating Default-initialized network with layers: {Layers}")

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)
            print(f"  Layer: {input_size} -> {output_size} (Default initialization)")

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = F.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define function to train model with progress tracking
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100, model_name="Model"):
    print(f"\nStarting training for {model_name}")
    print(f"Training for {epochs} epochs with learning rate {optimizer.param_groups[0]['lr']}")
    
    loss_accuracy = {'training_loss': [], 'validation_accuracy': []}  
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        batch_count = 0
        
        # Training phase
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            loss_accuracy['training_loss'].append(loss.item())
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in validation_loader:
                yhat = model(x.view(-1, 28 * 28))
                _, predicted = torch.max(yhat.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        loss_accuracy['validation_accuracy'].append(accuracy)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / batch_count
            print(f"  Epoch [{epoch+1:2d}/{epochs}] - Loss: {avg_loss:.4f}, "
                  f"Val Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    final_accuracy = loss_accuracy['validation_accuracy'][-1]
    print(f"  {model_name} training completed in {total_time:.2f}s")
    print(f"  Final validation accuracy: {final_accuracy:.2f}%\n")
    
    return loss_accuracy

# Create the datasets
print("Loading MNIST dataset...")
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())  # Fixed: train=False

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

# Create the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(validation_loader)}")

# Create the criterion function
criterion = nn.CrossEntropyLoss()
print(f"Using CrossEntropyLoss criterion")

# Create the parameters
input_dim = 28 * 28
output_dim = 10
layers = [input_dim, 100, 200, 100, output_dim]
learning_rate = 0.01
epochs = 30

print(f"\nNetwork Architecture: {layers}")
print(f"Learning rate: {learning_rate}")
print(f"Number of epochs: {epochs}")

# Train the model with the default initialization
print("\n" + "="*60)
print("TRAINING WITH DEFAULT INITIALIZATION")
print("="*60)
model = Net(layers)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=epochs, model_name="Default")

# Train the model with the He initialization
print("="*60)
print("TRAINING WITH HE INITIALIZATION")
print("="*60)
model_He = Net_He(layers)
optimizer = torch.optim.SGD(model_He.parameters(), lr=learning_rate)
training_results_He = train(model_He, criterion, train_loader, validation_loader, optimizer, epochs=epochs, model_name="He")

# Train the model with the Uniform initialization
print("="*60)
print("TRAINING WITH UNIFORM INITIALIZATION")
print("="*60)
model_Uniform = Net_Uniform(layers)
optimizer = torch.optim.SGD(model_Uniform.parameters(), lr=learning_rate)
training_results_Uniform = train(model_Uniform, criterion, train_loader, validation_loader, optimizer, epochs=epochs, model_name="Uniform")

# Plot the results
print("="*60)
print("CREATING VISUALIZATION")
print("="*60)

plt.figure(figsize=(15, 5))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(training_results_He['training_loss'], label='He', alpha=0.7)
plt.plot(training_results['training_loss'], label='Default', alpha=0.7)
plt.plot(training_results_Uniform['training_loss'], label='Uniform', alpha=0.7)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.title('Training Loss vs Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot the validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_results_He['validation_accuracy'], label='He', linewidth=2)
plt.plot(training_results['validation_accuracy'], label='Default', linewidth=2)
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform', linewidth=2)
plt.ylabel('Validation Accuracy (%)')
plt.xlabel('Epochs')
plt.title('Validation Accuracy vs Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary and analysis
print("="*60)
print("EXPERIMENT SUMMARY AND ANALYSIS")
print("="*60)

final_accuracies = {
    'He': training_results_He['validation_accuracy'][-1],
    'Default': training_results['validation_accuracy'][-1],
    'Uniform': training_results_Uniform['validation_accuracy'][-1]
}

print("\nFinal Validation Accuracies:")
for method, accuracy in final_accuracies.items():
    print(f"  {method:8s}: {accuracy:.2f}%")

best_method = max(final_accuracies, key=final_accuracies.get)
worst_method = min(final_accuracies, key=final_accuracies.get)

print(f"\nBest performing method: {best_method} ({final_accuracies[best_method]:.2f}%)")
print(f"Worst performing method: {worst_method} ({final_accuracies[worst_method]:.2f}%)")

# Calculate convergence speed (epochs to reach 90% accuracy)
convergence_analysis = {}
for name, results in [('He', training_results_He), ('Default', training_results), ('Uniform', training_results_Uniform)]:
    epochs_to_90 = None
    for i, acc in enumerate(results['validation_accuracy']):
        if acc >= 90.0:
            epochs_to_90 = i + 1
            break
    convergence_analysis[name] = epochs_to_90

print(f"\nConvergence Analysis (Epochs to reach 90% accuracy):")
for method, epochs in convergence_analysis.items():
    if epochs is not None:
        print(f"  {method:8s}: {epochs} epochs")
    else:
        print(f"  {method:8s}: Did not reach 90% accuracy")

print("\n" + "="*60)
print("KEY LESSONS FROM THIS EXPERIMENT")
print("="*60)

print("""
🔑 WEIGHT INITIALIZATION MATTERS - Here's what we learned:

1. **He Initialization (Kaiming)**: 
   - Specifically designed for ReLU activation functions
   - Sets weights based on the number of input neurons
   - Like a chef measuring ingredients based on pot size - scales appropriately
   - Usually performs best with ReLU networks

2. **Default PyTorch Initialization**:
   - Uses Xavier/Glorot initialization by default
   - Good general-purpose initialization
   - Like using a standard recipe - works well in most cases

3. **Uniform Initialization (0 to 1)**:
   - Weights are too large for deep networks
   - Can cause gradient problems (vanishing or exploding)
   - Like cooking with too much salt - throws off the whole dish

💡 **The Analogy**: Think of weight initialization like tuning a guitar:
   - He initialization: Each string tuned perfectly for its thickness
   - Default: Standard tuning that works for most songs  
   - Uniform: All strings tuned the same way (sounds terrible!)

📊 **What the graphs show**:
   - Training loss curves reveal how quickly each method learns
   - Validation accuracy shows which method generalizes best
   - Smoother curves typically indicate more stable training

🎯 **Practical takeaway**: 
   - Use He initialization for ReLU-based networks
   - Use Xavier/Glorot for tanh/sigmoid activations
   - Avoid uniform initialization for deep networks
   - Good initialization = faster training + better final performance
""")

print("Experiment completed successfully! 🎉")
