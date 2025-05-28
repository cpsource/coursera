# Test Uniform, Default and Xavier Uniform Initialization on MNIST dataset with tanh activation

# Import the libraries we need to use in this lab
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
import time

torch.manual_seed(0)

# Define the neural network with Xavier initialization
class Net_Xavier(nn.Module):
    def __init__(self, Layers):
        super(Net_Xavier, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define the neural network with Uniform initialization
class Net_Uniform(nn.Module):
    def __init__(self, Layers):
        super(Net_Uniform, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)
    
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define the neural network with Default initialization
class Net(nn.Module):
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)
    
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Function to Train the model
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100, model_name="Model"):
    print(f"\n=== Training {model_name} ===")
    start_time = time.time()
    
    loss_accuracy = {'training_loss': [], 'validation_accuracy': []}
    
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
        
        # Progress tracking
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / batch_count
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Val Accuracy: {accuracy:.2f}%, "
              f"Time: {epoch_time:.2f}s")
        
        # Early convergence check
        if epoch > 5 and accuracy > 95:
            print(f"Early convergence achieved at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    final_accuracy = loss_accuracy['validation_accuracy'][-1]
    print(f"{model_name} training completed in {total_time:.2f}s")
    print(f"Final validation accuracy: {final_accuracy:.2f}%")
    
    return loss_accuracy

# Data loading
print("=== Loading MNIST Dataset ===")
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Define criterion function
criterion = nn.CrossEntropyLoss()

# Set the parameters
input_dim = 28 * 28  # 784
output_dim = 10
layers = [input_dim, 100, 10, 100, 10, 100, output_dim]
epochs = 15
learning_rate = 0.01

print(f"\n=== Experiment Configuration ===")
print(f"Network architecture: {layers}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print(f"Batch size: {train_loader.batch_size}")

# Store results for comparison
all_results = {}

# Train the model with default initialization
print("\n" + "="*50)
model = Net(layers)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=epochs, model_name="Default Initialization")
all_results['Default'] = training_results

# Train the model with Xavier initialization
print("\n" + "="*50)
model_Xavier = Net_Xavier(layers)
optimizer = torch.optim.SGD(model_Xavier.parameters(), lr=learning_rate)
training_results_Xavier = train(model_Xavier, criterion, train_loader, validation_loader, optimizer, epochs=epochs, model_name="Xavier Initialization")
all_results['Xavier'] = training_results_Xavier

# Train the model with Uniform initialization  
print("\n" + "="*50)
model_Uniform = Net_Uniform(layers)
optimizer = torch.optim.SGD(model_Uniform.parameters(), lr=learning_rate)
training_results_Uniform = train(model_Uniform, criterion, train_loader, validation_loader, optimizer, epochs=epochs, model_name="Uniform Initialization")
all_results['Uniform'] = training_results_Uniform

# Plotting results
plt.figure(figsize=(15, 5))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(training_results_Xavier['training_loss'], label='Xavier', alpha=0.8)
plt.plot(training_results['training_loss'], label='Default', alpha=0.8)
plt.plot(training_results_Uniform['training_loss'], label='Uniform', alpha=0.8)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.title('Training Loss vs Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot the validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_results_Xavier['validation_accuracy'], label='Xavier', marker='o', alpha=0.8)
plt.plot(training_results['validation_accuracy'], label='Default', marker='s', alpha=0.8)
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform', marker='^', alpha=0.8)
plt.ylabel('Validation Accuracy (%)')
plt.xlabel('Epochs')
plt.title('Validation Accuracy vs Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary and conclusions
print("\n" + "="*60)
print("EXPERIMENT SUMMARY AND CONCLUSIONS")
print("="*60)

# Calculate final metrics
final_accuracies = {}
best_accuracies = {}
convergence_rates = {}

for name, results in all_results.items():
    final_accuracies[name] = results['validation_accuracy'][-1]
    best_accuracies[name] = max(results['validation_accuracy'])
    
    # Calculate epochs to reach 80% accuracy (convergence rate)
    convergence_epoch = None
    for i, acc in enumerate(results['validation_accuracy']):
        if acc >= 80.0:
            convergence_epoch = i + 1
            break
    convergence_rates[name] = convergence_epoch if convergence_epoch else len(results['validation_accuracy'])

print(f"\n📊 PERFORMANCE COMPARISON:")
print(f"{'Method':<20} {'Final Acc':<12} {'Best Acc':<12} {'Epochs to 80%':<15}")
print("-" * 60)
for name in ['Default', 'Xavier', 'Uniform']:
    print(f"{name:<20} {final_accuracies[name]:.2f}%{'':<7} {best_accuracies[name]:.2f}%{'':<7} {convergence_rates[name]:<15}")

# Determine the winner
best_method = max(final_accuracies.items(), key=lambda x: x[1])
fastest_method = min(convergence_rates.items(), key=lambda x: x[1])

print(f"\n🏆 WINNER: {best_method[0]} achieved the highest final accuracy ({best_method[1]:.2f}%)")
print(f"⚡ FASTEST CONVERGENCE: {fastest_method[0]} reached 80% accuracy in {fastest_method[1]} epochs")

print(f"\n💡 KEY INSIGHTS:")
print(f"1. Xavier initialization is designed to maintain gradient flow through deep networks")
print(f"2. Uniform initialization (0,1) can cause vanishing/exploding gradients with tanh")
print(f"3. Default PyTorch initialization provides a reasonable baseline")
print(f"4. The network architecture has alternating layer sizes which affects gradient propagation")

print(f"\n🔬 TECHNICAL ANALYSIS:")
if final_accuracies['Xavier'] > final_accuracies['Default']:
    print(f"- Xavier outperformed default initialization by {final_accuracies['Xavier'] - final_accuracies['Default']:.2f}%")
else:
    print(f"- Default initialization performed surprisingly well, matching Xavier's performance")

if final_accuracies['Uniform'] < 50:
    print(f"- Uniform initialization struggled due to poor weight scaling with tanh activation")
else:
    print(f"- Uniform initialization performed reasonably despite non-optimal weight scaling")

print(f"\n🎯 CONCLUSION:")
print(f"This experiment demonstrates the importance of proper weight initialization in deep learning.")
print(f"Xavier initialization is specifically designed for networks with tanh/sigmoid activations,")
print(f"helping maintain appropriate gradient magnitudes throughout training.")
