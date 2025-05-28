# Neural Networks with One Hidden Layer: Noisy XOR - FIXED VERSION

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

# Fixed plotting function for decision regions
def plot_decision_regions_2class(model, data_set, title=""):
    """Plot decision regions with proper model evaluation"""
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    model.eval()
    with torch.no_grad():
        mesh_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        predictions = model(mesh_points)
        
        # Handle both sigmoid and logits outputs
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(mesh_points)
        else:
            probs = predictions
            
        yhat = (probs[:, 0] > 0.5).numpy().reshape(xx.shape)
    
    # Plot decision regions
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light, alpha=0.8)
    
    # Plot data points
    plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c='red', marker='o', s=50, label='y=0', edgecolors='black')
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c='green', marker='s', s=50, label='y=1', edgecolors='black')
    
    plt.title(f"Decision Regions {title}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def accuracy(model, data_set):
    """Calculate accuracy with proper evaluation mode"""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(data_set.x)
        else:
            predictions = model(data_set.x)
        
        predicted_labels = (predictions[:, 0] > 0.5).float()
        true_labels = data_set.y[:, 0]
        return (predicted_labels == true_labels).float().mean().item()

# Original Net class with sigmoid (for comparison)
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# Improved Net class with logits for numerical stability
class NetWithLogits(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NetWithLogits, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)  # Raw logits, no final sigmoid
        return x
    
    def predict_proba(self, x):
        """Get probabilities for inference"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

# Fixed training function
def train(data_set, model, criterion, train_loader, optimizer, epochs=5, verbose=True):
    """Fixed training with proper gradient handling"""
    COST = []
    ACC = []
    
    for epoch in range(epochs):
        model.train()  # Set to training mode
        total = 0
        
        for x, y in train_loader:
            # Forward pass
            yhat = model(x)
            loss = criterion(yhat, y)
            
            # Backward pass (fixed: only one zero_grad call)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += loss.item()
        
        # Calculate accuracy
        acc = accuracy(model, data_set)
        ACC.append(acc)
        COST.append(total)
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {total:.4f}, Accuracy = {acc:.4f}")
    
    # Plot training curves
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.plot(COST, color=color, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(ACC, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Progress')
    fig.tight_layout()
    plt.show()
    
    return COST, ACC

# Fixed XOR Dataset class
class XOR_Data(Dataset):
    def __init__(self, N_s=100, noise_level=0.01):
        """
        Create XOR dataset with optional noise
        N_s: total number of samples (should be divisible by 4)
        noise_level: standard deviation of Gaussian noise
        """
        # Ensure N_s is divisible by 4 for equal class distribution
        N_s = (N_s // 4) * 4
        
        self.x = torch.zeros((N_s, 2))
        self.y = torch.zeros((N_s, 1))
        
        samples_per_class = N_s // 4
        
        # Create XOR pattern: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
        for i in range(samples_per_class):
            # Class 1: (0,0) -> 0
            self.x[i] = torch.tensor([0.0, 0.0])
            self.y[i] = torch.tensor([0.0])
            
            # Class 2: (0,1) -> 1
            self.x[i + samples_per_class] = torch.tensor([0.0, 1.0])
            self.y[i + samples_per_class] = torch.tensor([1.0])
            
            # Class 3: (1,0) -> 1
            self.x[i + 2*samples_per_class] = torch.tensor([1.0, 0.0])
            self.y[i + 2*samples_per_class] = torch.tensor([1.0])
            
            # Class 4: (1,1) -> 0
            self.x[i + 3*samples_per_class] = torch.tensor([1.0, 1.0])
            self.y[i + 3*samples_per_class] = torch.tensor([0.0])
        
        # Add noise once at the end
        if noise_level > 0:
            self.x += noise_level * torch.randn(N_s, 2)
        
        self.len = N_s
        
        print(f"Created XOR dataset: {N_s} samples, {samples_per_class} per class")
        print(f"Class distribution: {(self.y == 0).sum().item()} zeros, {(self.y == 1).sum().item()} ones")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def plot_stuff(self, title="XOR Dataset"):
        """Improved plotting function"""
        plt.figure(figsize=(8, 6))
        
        mask_0 = self.y[:, 0] == 0
        mask_1 = self.y[:, 0] == 1
        
        plt.scatter(self.x[mask_0, 0].numpy(), self.x[mask_0, 1].numpy(), 
                   c='red', marker='o', s=50, label="y=0", alpha=0.7, edgecolors='black')
        plt.scatter(self.x[mask_1, 0].numpy(), self.x[mask_1, 1].numpy(), 
                   c='green', marker='s', s=50, label="y=1", alpha=0.7, edgecolors='black')
        
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Function to compare different architectures
def compare_architectures(data_set, hidden_sizes, epochs=500, use_logits=False):
    """Compare different network architectures"""
    results = {}
    
    for h in hidden_sizes:
        print(f"\n{'='*60}")
        print(f"Training Network with {h} Hidden Neurons")
        print('='*60)
        
        # Choose model type
        if use_logits:
            model = NetWithLogits(2, h, 1)
            criterion = nn.BCEWithLogitsLoss()
            model_name = f"LogitsNet-{h}"
        else:
            model = Net(2, h, 1)
            criterion = nn.BCELoss()
            model_name = f"SigmoidNet-{h}"
        
        # Setup training
        learning_rate = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(dataset=data_set, batch_size=1)
        
        # Train model
        cost, acc = train(data_set, model, criterion, train_loader, optimizer, epochs=epochs)
        
        # Plot decision regions
        plot_decision_regions_2class(model, data_set, title=f"({h} hidden neurons)")
        
        # Store results
        results[model_name] = {
            'model': model,
            'final_loss': cost[-1],
            'final_accuracy': acc[-1],
            'cost_history': cost,
            'acc_history': acc
        }
        
        print(f"Final Results - Loss: {cost[-1]:.4f}, Accuracy: {acc[-1]:.4f}")
    
    return results

# Create and visualize dataset
print("Creating XOR Dataset...")
data_set = XOR_Data(N_s=100, noise_level=0.01)
data_set.plot_stuff()

# Compare different architectures with original approach
print("\n" + "="*80)
print("COMPARING ARCHITECTURES WITH SIGMOID + BCE LOSS")
print("="*80)

hidden_sizes = [1, 2, 3, 5]
results_sigmoid = compare_architectures(data_set, hidden_sizes, epochs=500, use_logits=False)

# Compare with improved approach
print("\n" + "="*80)
print("COMPARING ARCHITECTURES WITH LOGITS + BCEWITHLOGITSLOSS")
print("="*80)

results_logits = compare_architectures(data_set, hidden_sizes, epochs=500, use_logits=True)

# Final comparison plot
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot loss curves
axes[0, 0].set_title('Training Loss - Sigmoid Models')
for name, result in results_sigmoid.items():
    axes[0, 0].plot(result['cost_history'], label=name)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title('Training Loss - Logits Models')
for name, result in results_logits.items():
    axes[0, 1].plot(result['cost_history'], label=name)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot accuracy curves
axes[1, 0].set_title('Training Accuracy - Sigmoid Models')
for name, result in results_sigmoid.items():
    axes[1, 0].plot(result['acc_history'], label=name)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].set_title('Training Accuracy - Logits Models')
for name, result in results_logits.items():
    axes[1, 1].plot(result['acc_history'], label=name)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table
print("\nSUMMARY TABLE:")
print("-" * 80)
print(f"{'Model':<15} {'Final Loss':<12} {'Final Accuracy':<15} {'Converged':<10}")
print("-" * 80)

all_results = {**results_sigmoid, **results_logits}
for name, result in all_results.items():
    converged = "Yes" if result['final_accuracy'] > 0.95 else "No"
    print(f"{name:<15} {result['final_loss']:<12.4f} {result['final_accuracy']:<15.4f} {converged:<10}")

print("\nKey Insights:")
print("• XOR requires at least 2 hidden neurons to be solvable")
print("• More neurons generally lead to better convergence")
print("• BCEWithLogitsLoss tends to be more numerically stable")
print("• The XOR problem demonstrates why linear models fail on non-linearly separable data")
