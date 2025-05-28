# Using Dropout for Classification - Fixed and Enhanced Version

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# The function for plotting the diagram
def plot_decision_regions_3class(data_set, model=None, title_suffix=""):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    newdata = np.c_[xx.ravel(), yy.ravel()]

    # True decision boundary
    Z = data_set.multi_dim_poly(newdata).flatten()
    f = np.zeros(Z.shape)
    f[Z > 0] = 1
    f = f.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    
    if model is not None:
        model.eval()  # Ensure model is in evaluation mode
        XX = torch.Tensor(newdata)
        with torch.no_grad():  # No gradients needed for inference
            _, yhat = torch.max(model(XX), 1)
        yhat = yhat.numpy().reshape(xx.shape)
        plt.pcolormesh(xx, yy, yhat, cmap=cmap_light, alpha=0.8)
        # Create contour and manually add to legend
        contour_lines = plt.contour(xx, yy, f, colors='black', linestyles='--', linewidths=2)
        # Add manual legend entry for true boundary
        plt.plot([], [], 'k--', linewidth=2, label='True boundary')
    else:
        plt.contour(xx, yy, f, colors='black', linestyles='-', linewidths=2)
        plt.pcolormesh(xx, yy, f, cmap=cmap_light, alpha=0.8)

    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', s=50, label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', s=50, label='Class 1')
    
    plt.title(f"Decision Regions vs True Boundary{title_suffix}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.show()

# The function for calculating accuracy
def accuracy(model, data_set):
    model.eval()
    with torch.no_grad():
        _, yhat = torch.max(model(data_set.x), 1)
        acc = (yhat == data_set.y).float().mean().item()
    return acc

# Enhanced Data class with better documentation and bug fixes
class Data(Dataset):
    """
    Dataset class that generates a 2D binary classification problem.
    
    Think of this like creating a curved boundary in 2D space (like a wavy line),
    and then adding noise to make the problem more realistic.
    """

    def __init__(self, N_SAMPLES=1000, noise_std=0.15, train=True):
        print(f"Creating dataset with {N_SAMPLES} samples, noise_std={noise_std}, train={train}")
        
        # Coefficients for polynomial decision boundary
        # This creates a curved boundary in 2D space
        a = np.array([-1, 1, 2, 1, 1, -3, 1]).reshape(-1, 1)
        
        # Generate random 2D points
        self.x = np.random.rand(N_SAMPLES, 2)
        
        # Calculate polynomial function value for each point
        # f(x,y) = a0 + a1*x + a2*y + a3 + a4*x*y + a5*x^2 + a6*y^2
        self.f = (a[0] + 
                 self.x @ a[1:3] + 
                 (self.x[:, 0:1] * self.x[:, 1:2]) * a[4] + 
                 (self.x ** 2) @ a[5:7]).flatten()
        
        self.a = a

        # Create binary labels based on sign of polynomial
        self.y = np.zeros(N_SAMPLES, dtype=int)
        self.y[self.f > 0] = 1
        
        # Convert to PyTorch tensors
        self.y = torch.from_numpy(self.y).long()
        self.x = torch.from_numpy(self.x).float()
        
        # Add noise to make problem more realistic
        if train:
            self.x = self.x + noise_std * torch.randn(self.x.size())
            print(f"Added training noise with std={noise_std}")
        
        self.f = torch.from_numpy(self.f)
        self.len = N_SAMPLES  # Fix: was missing this attribute
        
        print(f"Dataset created: {len(self)} samples")
        print(f"Class distribution: {(self.y == 0).sum().item()} class 0, {(self.y == 1).sum().item()} class 1")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def plot(self):
        """Plot the dataset with true decision boundary"""
        X = self.x.numpy()
        y = self.y.numpy()
        h = .02
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = self.multi_dim_poly(np.c_[xx.ravel(), yy.ravel()]).flatten()
        f = np.zeros(Z.shape)
        f[Z > 0] = 1
        f = f.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.title('True Decision Boundary and Sample Points with Noise')
        plt.contour(xx, yy, f, colors='black', linestyles='-', linewidths=2)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', s=50, label='Class 0', alpha=0.7)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', s=50, label='Class 1', alpha=0.7)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def multi_dim_poly(self, x):
        """Calculate polynomial function value for given points"""
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Same polynomial as in __init__
        out = (self.a[0] + 
               x @ self.a[1:3] + 
               (x[:, 0:1] * x[:, 1:2]) * self.a[4] + 
               (x ** 2) @ self.a[5:7]).flatten()
        return out

# Enhanced Neural Network with better dropout explanation
class Net(nn.Module):
    """
    Neural Network with optional dropout.
    
    Dropout is like temporarily removing random neurons during training.
    Think of it as forcing the network to not rely too heavily on any single neuron,
    making it more robust and reducing overfitting.
    """

    def __init__(self, in_size, n_hidden, out_size, p=0):
        super(Net, self).__init__()
        self.dropout_prob = p
        print(f"Creating network: {in_size} -> {n_hidden} -> {n_hidden} -> {out_size}")
        print(f"Dropout probability: {p} ({'Enabled' if p > 0 else 'Disabled'})")
        
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

    def forward(self, x):
        # First hidden layer with dropout
        x = self.linear1(x)
        x = F.relu(x)
        x = self.drop(x)  # Dropout after activation
        
        # Second hidden layer with dropout
        x = self.linear2(x)
        x = F.relu(x)
        x = self.drop(x)  # Dropout after activation
        
        # Output layer (no dropout here)
        x = self.linear3(x)
        return x

def print_dropout_effect(model, sample_input, num_passes=5):
    """
    Demonstrate how dropout creates different outputs during training.
    This shows the randomness that dropout introduces.
    """
    print(f"\n--- Dropout Effect Demonstration ---")
    print(f"Dropout probability: {model.dropout_prob}")
    
    if model.dropout_prob == 0:
        print("No dropout - output will be consistent")
    else:
        print(f"With dropout - output will vary due to random neuron masking")
    
    model.train()  # Enable dropout
    
    for i in range(num_passes):
        with torch.no_grad():
            output = model(sample_input)
            print(f"Pass {i+1}: {output[0].numpy()}")
    
    print("This randomness during training helps prevent overfitting!")

# Create datasets
print("=== Creating Datasets ===")
data_set = Data(N_SAMPLES=1000, noise_std=0.2, train=True)
validation_set = Data(N_SAMPLES=500, noise_std=0.2, train=False)

# Visualize the data
print("\n=== Visualizing Training Data ===")
data_set.plot()

# Create models
print("\n=== Creating Models ===")
model_no_dropout = Net(2, 300, 2, p=0.0)
model_with_dropout = Net(2, 300, 2, p=0.5)

# Demonstrate dropout effect
print("\n=== Demonstrating Dropout Effect ===")
sample_input = data_set.x[:1]  # First sample
print_dropout_effect(model_no_dropout, sample_input)
print_dropout_effect(model_with_dropout, sample_input)

# Set up training
print("\n=== Setting Up Training ===")
optimizer_no_dropout = torch.optim.Adam(model_no_dropout.parameters(), lr=0.01)
optimizer_with_dropout = torch.optim.Adam(model_with_dropout.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Initialize loss tracking
LOSS = {
    'training_no_dropout': [],
    'validation_no_dropout': [],
    'training_with_dropout': [],
    'validation_with_dropout': []
}

# Training function with better instrumentation
def train_model(epochs):
    print(f"\n=== Training for {epochs} epochs ===")
    
    for epoch in range(epochs):
        # Set models to training mode
        model_no_dropout.train()
        model_with_dropout.train()
        
        # Forward pass for both models
        yhat_no_dropout = model_no_dropout(data_set.x)
        yhat_with_dropout = model_with_dropout(data_set.x)
        
        # Calculate training losses
        loss_no_dropout = criterion(yhat_no_dropout, data_set.y)
        loss_with_dropout = criterion(yhat_with_dropout, data_set.y)
        
        # Store training losses
        LOSS['training_no_dropout'].append(loss_no_dropout.item())
        LOSS['training_with_dropout'].append(loss_with_dropout.item())
        
        # Calculate validation losses (models in eval mode)
        model_no_dropout.eval()
        model_with_dropout.eval()
        
        with torch.no_grad():
            val_pred_no_dropout = model_no_dropout(validation_set.x)
            val_pred_with_dropout = model_with_dropout(validation_set.x)
            
            val_loss_no_dropout = criterion(val_pred_no_dropout, validation_set.y)
            val_loss_with_dropout = criterion(val_pred_with_dropout, validation_set.y)
            
            LOSS['validation_no_dropout'].append(val_loss_no_dropout.item())
            LOSS['validation_with_dropout'].append(val_loss_with_dropout.item())
        
        # Backward pass and optimization
        optimizer_no_dropout.zero_grad()
        optimizer_with_dropout.zero_grad()
        
        loss_no_dropout.backward()
        loss_with_dropout.backward()
        
        optimizer_no_dropout.step()
        optimizer_with_dropout.step()
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            train_acc_no_dropout = accuracy(model_no_dropout, data_set)
            train_acc_with_dropout = accuracy(model_with_dropout, data_set)
            val_acc_no_dropout = accuracy(model_no_dropout, validation_set)
            val_acc_with_dropout = accuracy(model_with_dropout, validation_set)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  No Dropout    - Train Loss: {loss_no_dropout.item():.4f}, Train Acc: {train_acc_no_dropout:.4f}, Val Acc: {val_acc_no_dropout:.4f}")
            print(f"  With Dropout  - Train Loss: {loss_with_dropout.item():.4f}, Train Acc: {train_acc_with_dropout:.4f}, Val Acc: {val_acc_with_dropout:.4f}")
            print(f"  Gap (Overfitting indicator): Train-Val = {train_acc_no_dropout - val_acc_no_dropout:.4f} (no dropout), {train_acc_with_dropout - val_acc_with_dropout:.4f} (with dropout)")

# Train the models
epochs = 500
train_model(epochs)

# Final evaluation
print("\n=== Final Evaluation ===")
model_no_dropout.eval()
model_with_dropout.eval()

train_acc_no_dropout = accuracy(model_no_dropout, data_set)
train_acc_with_dropout = accuracy(model_with_dropout, data_set)
val_acc_no_dropout = accuracy(model_no_dropout, validation_set)
val_acc_with_dropout = accuracy(model_with_dropout, validation_set)

print(f"Final Results:")
print(f"  Model WITHOUT Dropout:")
print(f"    Training Accuracy:   {train_acc_no_dropout:.4f}")
print(f"    Validation Accuracy: {val_acc_no_dropout:.4f}")
print(f"    Overfitting Gap:     {train_acc_no_dropout - val_acc_no_dropout:.4f}")

print(f"\n  Model WITH Dropout:")
print(f"    Training Accuracy:   {train_acc_with_dropout:.4f}")
print(f"    Validation Accuracy: {val_acc_with_dropout:.4f}")
print(f"    Overfitting Gap:     {train_acc_with_dropout - val_acc_with_dropout:.4f}")

print(f"\n  Dropout Benefits:")
print(f"    Reduced Overfitting: {(train_acc_no_dropout - val_acc_no_dropout) - (train_acc_with_dropout - val_acc_with_dropout):.4f}")
print(f"    Validation Improvement: {val_acc_with_dropout - val_acc_no_dropout:.4f}")

# Visualize decision boundaries
print("\n=== Visualizing Decision Boundaries ===")
plot_decision_regions_3class(validation_set, None, " - True Boundary Only")
plot_decision_regions_3class(validation_set, model_no_dropout, " - Model WITHOUT Dropout")
plot_decision_regions_3class(validation_set, model_with_dropout, " - Model WITH Dropout")

# Plot training curves
print("\n=== Plotting Training Curves ===")
plt.figure(figsize=(15, 5))

# Loss curves
plt.subplot(1, 2, 1)
epochs_range = range(1, len(LOSS['training_no_dropout']) + 1)

plt.plot(epochs_range, LOSS['training_no_dropout'], 'b-', label='Training (No Dropout)', linewidth=2)
plt.plot(epochs_range, LOSS['validation_no_dropout'], 'b--', label='Validation (No Dropout)', linewidth=2)
plt.plot(epochs_range, LOSS['training_with_dropout'], 'r-', label='Training (With Dropout)', linewidth=2)
plt.plot(epochs_range, LOSS['validation_with_dropout'], 'r--', label='Validation (With Dropout)', linewidth=2)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Log scale plot for better visualization
plt.subplot(1, 2, 2)
plt.plot(epochs_range, np.log(LOSS['training_no_dropout']), 'b-', label='Training (No Dropout)', linewidth=2)
plt.plot(epochs_range, np.log(LOSS['validation_no_dropout']), 'b--', label='Validation (No Dropout)', linewidth=2)
plt.plot(epochs_range, np.log(LOSS['training_with_dropout']), 'r-', label='Training (With Dropout)', linewidth=2)
plt.plot(epochs_range, np.log(LOSS['validation_with_dropout']), 'r--', label='Validation (With Dropout)', linewidth=2)

plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Training and Validation Loss (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Understanding Dropout ===")
print("""
DROPOUT EXPLAINED:

1. WHAT IS DROPOUT?
   - During training, dropout randomly sets some neurons to zero
   - Each neuron has a probability 'p' of being turned off
   - This happens independently for each training example

2. WHY DOES IT HELP?
   - Prevents co-adaptation: neurons can't rely too heavily on specific other neurons
   - Forces the network to learn more robust features
   - Acts as regularization, reducing overfitting

3. TRAINING vs INFERENCE:
   - During training: dropout is active (neurons randomly turned off)
   - During inference: dropout is turned off (all neurons active)
   - Outputs are scaled appropriately to maintain consistency

4. THE ANALOGY:
   Think of dropout like training a sports team where random players sit out each game.
   This forces the remaining players to adapt and work together better,
   making the whole team more resilient when everyone plays together.

5. KEY OBSERVATIONS FROM THIS EXPERIMENT:
   - Model with dropout may have slightly lower training accuracy
   - But it should have BETTER validation accuracy (less overfitting)
   - The gap between training and validation accuracy should be smaller
""")
