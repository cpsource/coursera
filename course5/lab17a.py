# Neural Networks with Momentum - Fixed Version

# Import the libraries for this lab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
np.random.seed(1)

print("=" * 60)
print("NEURAL NETWORKS WITH MOMENTUM - LAB ANALYSIS")
print("=" * 60)
print("Libraries imported and random seeds set for reproducibility\n")

# Define a function for plot the decision region
def plot_decision_regions_3class(model, data_set, title_suffix=""):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:]==0, 0], X[y[:]==0, 1], 'ro', label='y=0')
    plt.plot(X[y[:]==1, 0], X[y[:]==1, 1], 'go', label='y=1')
    plt.plot(X[y[:]==2, 0], X[y[:]==2, 1], 'bo', label='y=2')  # Fixed: was 'o'
    plt.title(f"Decision Region {title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Create the dataset class
class Data(Dataset):
    # Constructor
    def __init__(self, K=3, N=500):
        print(f"Creating dataset with {K} classes and {N} samples per class...")
        D = 2
        X = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K, dtype='uint8')  # class labels
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j

        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]
        print(f"Dataset created: {self.len} total samples")

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the diagram
    def plot_data(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'ro', label="y=0", alpha=0.7)
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'go', label="y=1", alpha=0.7)
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'bo', label="y=2", alpha=0.7)
        plt.title("Training Data - Spiral Dataset")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Neural Network Module
class Net(nn.Module):
    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))
        
        # Print network architecture
        print(f"Network architecture: {' -> '.join(map(str, Layers))}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation

# Define the function for training the model
def train(data_set, model, criterion, train_loader, optimizer, epochs=100, momentum_val=0):
    print(f"\nTraining model with momentum = {momentum_val}")
    print(f"Epochs: {epochs}, Batch size: {train_loader.batch_size}")
    print("-" * 40)
    
    LOSS = []
    ACC = []
    
    # Print progress every 20 epochs
    progress_intervals = [19, 39, 59, 79, 99]  # 0-indexed
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        model.train()  # Set model to training mode
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for this epoch
        avg_loss = epoch_loss / num_batches
        LOSS.append(avg_loss)
        
        # Calculate accuracy
        current_acc = accuracy(model, data_set)
        ACC.append(current_acc)
        
        # Print progress at specified intervals
        if epoch in progress_intervals:
            print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Accuracy = {current_acc:.4f}")
    
    print(f"Final: Loss = {LOSS[-1]:.4f}, Accuracy = {ACC[-1]:.4f}")
    
    # Plot training progress
    results = {"Loss": LOSS, "Accuracy": ACC}
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.plot(LOSS, color=color, linewidth=2)
    ax1.set_xlabel('Epoch', color='black')
    ax1.set_ylabel('Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(ACC, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Training Progress (Momentum = {momentum_val})')
    fig.tight_layout()
    plt.show()
    
    return results

# Define a function for calculating accuracy
def accuracy(model, data_set):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).float().mean().item()  # Fixed: convert to float for mean

# Create the dataset and plot it
print("Creating and visualizing dataset...")
data_set = Data()
data_set.plot_data()

# Initialize a dictionary to contain the cost and accuracy
momentum_values = [0, 0.1, 0.2, 0.4, 0.5]
Results = {}

print("\n" + "=" * 60)
print("TRAINING MODELS WITH DIFFERENT MOMENTUM VALUES")
print("=" * 60)

# Train models with different momentum values
for momentum in momentum_values:
    print(f"\n{'='*20} MOMENTUM = {momentum} {'='*20}")
    
    # Create new model for each momentum value
    Layers = [2, 50, 3]
    model = Net(Layers)
    learning_rate = 0.10
    
    # Create optimizer with specified momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_loader = DataLoader(dataset=data_set, batch_size=20)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    Results[f"momentum {momentum}"] = train(data_set, model, criterion, train_loader, 
                                          optimizer, epochs=100, momentum_val=momentum)
    
    # Plot decision regions
    plot_decision_regions_3class(model, data_set, f"(Momentum = {momentum})")
    
    print(f"Completed training for momentum = {momentum}")

print("\n" + "=" * 60)
print("COMPARING RESULTS ACROSS DIFFERENT MOMENTUM VALUES")
print("=" * 60)

# Compare Results of Different Momentum Terms
print("\nFinal Results Summary:")
print("-" * 50)
for key, value in Results.items():
    final_loss = value['Loss'][-1]
    final_acc = value['Accuracy'][-1]
    print(f"{key:15}: Final Loss = {final_loss:.4f}, Final Accuracy = {final_acc:.4f}")

# Plot the Loss result for each momentum value
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for key, value in Results.items():
    plt.plot(value['Loss'], label=key, linewidth=2)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison Across Momentum Values')
plt.grid(True, alpha=0.3)

# Plot the Accuracy result for each momentum value
plt.subplot(1, 2, 2)
for key, value in Results.items():
    plt.plot(value['Accuracy'], label=key, linewidth=2)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Across Momentum Values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ANALYSIS AND SUMMARY")
print("=" * 60)

print("""
MOMENTUM IN NEURAL NETWORKS - DETAILED EXPLANATION:

Think of momentum like a ball rolling down a hill. Without momentum, the ball stops 
and changes direction immediately when it hits a small bump. With momentum, the ball 
keeps some of its previous motion, helping it roll over small bumps and reach the 
bottom faster.

In neural networks:
- Momentum helps the optimizer remember previous gradient directions
- It smooths out oscillations in the loss landscape
- It can help escape local minima and reach better solutions faster

HOW MOMENTUM WORKS:
The momentum parameter (β) controls how much of the previous update to remember:
- β = 0: No momentum (standard SGD)
- β = 0.1: Remember 10% of previous direction
- β = 0.9: Remember 90% of previous direction (common choice)

MATHEMATICAL ANALOGY:
Like a weighted average of directions:
New_direction = β × Previous_direction + (1-β) × Current_gradient

RESULTS ANALYSIS:
""")

# Analyze the results
best_momentum = min(Results.keys(), key=lambda k: Results[k]['Loss'][-1])
worst_momentum = max(Results.keys(), key=lambda k: Results[k]['Loss'][-1])

print(f"Best performing momentum: {best_momentum}")
print(f"  - Final Loss: {Results[best_momentum]['Loss'][-1]:.4f}")
print(f"  - Final Accuracy: {Results[best_momentum]['Accuracy'][-1]:.4f}")

print(f"\nWorst performing momentum: {worst_momentum}")
print(f"  - Final Loss: {Results[worst_momentum]['Loss'][-1]:.4f}")
print(f"  - Final Accuracy: {Results[worst_momentum]['Accuracy'][-1]:.4f}")

print(f"""
OBSERVATIONS:
1. The spiral dataset is a complex non-linear classification problem
2. Different momentum values show different convergence behaviors
3. Too little momentum may lead to slow convergence
4. Too much momentum may cause overshooting and instability

PRACTICAL TAKEAWAYS:
- Momentum typically helps neural network training
- Common momentum values are between 0.8 and 0.95
- The optimal momentum depends on your specific problem
- Always experiment with different values for your dataset

This lab demonstrates how momentum affects the training dynamics of neural networks
on a challenging 3-class spiral classification problem.
""")

print("=" * 60)
print("LAB COMPLETED SUCCESSFULLY!")
print("=" * 60)
