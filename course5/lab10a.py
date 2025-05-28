# Deeper Neural Networks with nn.ModuleList()
# Cleaned and improved version with comprehensive logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

# Set seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

print("=" * 60)
print("NEURAL NETWORK LAB - Starting Setup")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print()

def plot_decision_regions_3class(model, data_set, title="Decision Regions"):
    """Plot decision boundaries for 3-class classification"""
    print(f"📊 Plotting decision regions: {title}")
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create prediction grid
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light, alpha=0.8)
    
    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', s=50, label='Class 0', edgecolors='black')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green', marker='s', s=50, label='Class 1', edgecolors='black')
    plt.scatter(X[y == 2, 0], X[y == 2, 1], c='blue', marker='^', s=50, label='Class 2', edgecolors='black')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

class Data(Dataset):
    """
    Create spiral dataset for 3-class classification
    Think of this like creating three intertwined spirals - like DNA strands twisted together
    """
    
    def __init__(self, K=3, N=500):
        print(f"🔄 Creating dataset with {K} classes, {N} samples per class")
        
        D = 2  # 2D features
        X = np.zeros((N * K, D))  # Feature matrix
        y = np.zeros(N * K, dtype='uint8')  # Labels
        
        for j in range(K):
            print(f"   Creating class {j} spiral...")
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius from center
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # angle with noise
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]
        
        print(f"✅ Dataset created: {self.len} total samples")
        print(f"   Feature shape: {self.x.shape}")
        print(f"   Label shape: {self.y.shape}")
        print(f"   Classes: {torch.unique(self.y).tolist()}")
        print()
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
    def plot_data(self):
        """Plot the raw dataset"""
        print("📊 Plotting original dataset...")
        plt.figure(figsize=(10, 8))
        
        colors = ['red', 'green', 'blue']
        markers = ['o', 's', '^']
        
        for class_idx in range(3):
            mask = self.y == class_idx
            plt.scatter(self.x[mask, 0].numpy(), self.x[mask, 1].numpy(), 
                       c=colors[class_idx], marker=markers[class_idx], 
                       s=50, label=f"Class {class_idx}", alpha=0.7, edgecolors='black')
        
        plt.title("Original Dataset - Three Spiral Classes", fontsize=14, fontweight='bold')
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class Net(nn.Module):
    """
    Neural Network with configurable hidden layers
    Think of this like a stack of pancakes - each layer transforms the data
    """
    
    def __init__(self, layers):
        super(Net, self).__init__()
        print(f"🏗️  Building neural network with architecture: {layers}")
        
        self.hidden = nn.ModuleList()
        self.layer_sizes = layers
        
        # Create layers based on the input architecture
        for i, (input_size, output_size) in enumerate(zip(layers, layers[1:])):
            layer = nn.Linear(input_size, output_size)
            self.hidden.append(layer)
            print(f"   Layer {i+1}: {input_size} → {output_size}")
        
        print(f"✅ Network created with {len(self.hidden)} layers")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total parameters: {total_params:,}")
        print()
    
    def forward(self, x):
        """Forward pass through the network"""
        activation = x
        L = len(self.hidden)
        
        for layer_idx, linear_transform in enumerate(self.hidden):
            if layer_idx < L - 1:  # Hidden layers use ReLU
                activation = F.relu(linear_transform(activation))
            else:  # Output layer (no activation)
                activation = linear_transform(activation)
        
        return activation

def train_model(data_set, model, criterion, train_loader, optimizer, epochs=100, print_every=10):
    """
    Train the neural network with detailed logging
    Think of this like teaching a student - we show examples repeatedly and track progress
    """
    print(f"🎯 Starting training for {epochs} epochs...")
    print(f"   Optimizer: {type(optimizer).__name__}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Loss function: {type(criterion).__name__}")
    print()
    
    LOSS = []
    ACC = []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()  # Set to training mode
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Forward pass
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            LOSS.append(loss.item())
        
        # Calculate accuracy for this epoch
        current_accuracy = calculate_accuracy(model, data_set)
        ACC.append(current_accuracy)
        
        # Track best accuracy
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch [{epoch+1:4d}/{epochs}] | "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Accuracy: {current_accuracy:.4f} | "
                  f"Best: {best_accuracy:.4f}")
    
    print(f"\n✅ Training completed!")
    print(f"   Final accuracy: {ACC[-1]:.4f}")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    print(f"   Total iterations: {len(LOSS):,}")
    print()
    
    # Plot training curves
    plot_training_curves(LOSS, ACC)
    
    return LOSS, ACC

def calculate_accuracy(model, data_set):
    """Calculate model accuracy on the dataset"""
    model.eval()
    with torch.no_grad():
        outputs = model(data_set.x)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == data_set.y).float().mean().item()
    return accuracy

def plot_training_curves(losses, accuracies):
    """Plot loss and accuracy curves"""
    print("📊 Plotting training curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(losses, color='red', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='red')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='red')
    
    # Accuracy plot  
    epochs = range(1, len(accuracies) + 1)
    ax2.plot(epochs, accuracies, color='blue', linewidth=2, marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.set_title('Training Accuracy Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def run_experiment(layers, learning_rate, epochs, data_set, experiment_name):
    """Run a complete training experiment"""
    print("=" * 80)
    print(f"🧪 EXPERIMENT: {experiment_name}")
    print("=" * 80)
    
    # Create model
    model = Net(layers)
    
    # Setup training
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset=data_set, batch_size=20, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    losses, accuracies = train_model(data_set, model, criterion, train_loader, optimizer, epochs)
    
    # Plot results
    plot_decision_regions_3class(model, data_set, f"{experiment_name} - Decision Boundaries")
    
    return model, losses, accuracies

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create dataset
    print("🔧 SETUP PHASE")
    print("-" * 40)
    data_set = Data(K=3, N=500)
    data_set.plot_data()
    
    # Experiment 1: Simple network (1 hidden layer)
    model1, loss1, acc1 = run_experiment(
        layers=[2, 50, 3],
        learning_rate=0.10,
        epochs=100,
        data_set=data_set,
        experiment_name="Single Hidden Layer (50 neurons)"
    )
    
    # Experiment 2: Deeper network (2 hidden layers)
    model2, loss2, acc2 = run_experiment(
        layers=[2, 10, 10, 3],
        learning_rate=0.01,
        epochs=1000,
        data_set=data_set,
        experiment_name="Two Hidden Layers (10+10 neurons)"
    )
    
    # Experiment 3: Even deeper network (3 hidden layers)
    model3, loss3, acc3 = run_experiment(
        layers=[2, 10, 10, 10, 3],
        learning_rate=0.01,
        epochs=1000,
        data_set=data_set,
        experiment_name="Three Hidden Layers (10+10+10 neurons)"
    )
    
    # Final comparison
    print("=" * 80)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Experiment 1 - Final Accuracy: {acc1[-1]:.4f}")
    print(f"Experiment 2 - Final Accuracy: {acc2[-1]:.4f}")
    print(f"Experiment 3 - Final Accuracy: {acc3[-1]:.4f}")
    print("=" * 80)
