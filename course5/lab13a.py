# Neural Network with Weight Initialization Analysis
# This code demonstrates the importance of proper weight initialization

import torch
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
print("🔧 Setting random seed for reproducible results")

def PlotStuff(X, Y, model, epoch, leg=True):
    """Plot model predictions vs actual data"""
    plt.figure(figsize=(10, 6))
    plt.plot(X.numpy(), model(X).detach().numpy(), label=f'Model Prediction (Epoch {epoch})', linewidth=2)
    plt.plot(X.numpy(), Y.numpy(), 'r-', label='True Function', linewidth=2)
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.title(f'Model Performance at Epoch {epoch}')
    plt.grid(True, alpha=0.3)
    if leg:
        plt.legend()
    plt.show()

class Net(nn.Module):
    """
    Neural Network Architecture:
    Input → Linear(1→2) → Sigmoid → Linear(2→1) → Sigmoid → Output
    
    Think of this like a data processing pipeline:
    - First layer: Transforms 1D input into 2D hidden representation
    - Second layer: Combines hidden features into final prediction
    """
    
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        print(f"🏗️  Building network: {D_in} → {H} → {D_out}")
        
        # Hidden layer (like feature extractors)
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
        # Store intermediate values for analysis
        self.a1 = None  # First layer activations
        self.l1 = None  # First layer linear output
        self.l2 = None  # Second layer linear output
    
    def forward(self, x):
        """Forward pass through the network"""
        # First layer: linear transformation + sigmoid activation
        self.l1 = self.linear1(x)
        self.a1 = sigmoid(self.l1)
        
        # Second layer: linear transformation + sigmoid activation
        self.l2 = self.linear2(self.a1)
        yhat = sigmoid(self.l2)  # Final prediction
        
        return yhat
    
    def print_weights(self, description=""):
        """Print current weights and biases for analysis"""
        print(f"\n📊 {description}")
        print(f"Layer 1 weights: {self.linear1.weight.data.numpy()}")
        print(f"Layer 1 biases:  {self.linear1.bias.data.numpy()}")
        print(f"Layer 2 weights: {self.linear2.weight.data.numpy()}")
        print(f"Layer 2 biases:  {self.linear2.bias.data.numpy()}")

def train(Y, X, model, optimizer, criterion, epochs=1000):
    """
    Training loop with detailed monitoring
    Like teaching a student - we show examples and adjust based on mistakes
    """
    print(f"\n🎓 Starting training for {epochs} epochs...")
    cost = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Process each training example
        for y, x in zip(Y, X):
            # Forward pass: make prediction
            yhat = model(x)
            
            # Calculate loss: how wrong were we?
            loss = criterion(yhat, y)
            
            # Backward pass: calculate gradients
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Calculate new gradients
            optimizer.step()       # Update weights
            
            total_loss += loss.item()
        
        cost.append(total_loss)
        
        # Periodic monitoring
        if epoch % 300 == 0:
            print(f"\n📈 Epoch {epoch}: Total Loss = {total_loss:.4f}")
            
            # Show model performance
            PlotStuff(X, Y, model, epoch, leg=True)
            
            # Visualize hidden layer activations
            # This shows how the network internally represents the data
            model(X)  # Forward pass to compute activations
            if model.a1.shape[1] >= 2:  # If we have at least 2 hidden units
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(
                    model.a1.detach().numpy()[:, 0], 
                    model.a1.detach().numpy()[:, 1], 
                    c=Y.numpy().reshape(-1),
                    cmap='coolwarm',
                    alpha=0.7
                )
                plt.colorbar(scatter, label='True Labels')
                plt.xlabel('Hidden Unit 1 Activation')
                plt.ylabel('Hidden Unit 2 Activation')
                plt.title(f'Hidden Layer Activations (Epoch {epoch})')
                plt.grid(True, alpha=0.3)
                plt.show()
            
            # Show weight evolution
            model.print_weights(f"Weights at Epoch {epoch}")
    
    print(f"\n✅ Training completed!")
    return cost

def create_data():
    """
    Create training data: a step function
    Like a light switch - off for x < -4 or x > 4, on for -4 ≤ x ≤ 4
    """
    print("📊 Creating training data...")
    X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
    Y = torch.zeros(X.shape[0])
    Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Y values: {torch.unique(Y)} (0=off, 1=on)")
    
    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.plot(X.numpy(), Y.numpy(), 'ro-', linewidth=2, markersize=4)
    plt.xlabel('Input (x)')
    plt.ylabel('Target Output')
    plt.title('Training Data: Step Function')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return X, Y

def criterion_cross(outputs, labels):
    """
    Cross-entropy loss function
    Measures how far our predictions are from the true labels
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    outputs = torch.clamp(outputs, epsilon, 1 - epsilon)
    
    loss = -torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return loss

def initialize_weights_same(model):
    """Initialize all weights to the same values - this causes problems!"""
    print("\n⚠️  WARNING: Initializing all weights to SAME values")
    print("This is like having identical twins trying to learn different skills!")
    
    with torch.no_grad():
        model.linear1.weight.fill_(1.0)  # All weights = 1.0
        model.linear1.bias.fill_(0.0)    # All biases = 0.0
        model.linear2.weight.fill_(1.0)
        model.linear2.bias.fill_(0.0)
    
    model.print_weights("Initial weights (SAME - problematic)")

def initialize_weights_random(model):
    """Initialize weights randomly - this is better!"""
    print("\n✅ Using random weight initialization")
    print("This gives each neuron a unique starting point!")
    
    # PyTorch's default initialization is already good, but let's be explicit
    with torch.no_grad():
        nn.init.xavier_uniform_(model.linear1.weight)
        nn.init.zeros_(model.linear1.bias)
        nn.init.xavier_uniform_(model.linear2.weight)
        nn.init.zeros_(model.linear2.bias)
    
    model.print_weights("Initial weights (RANDOM - good)")

def test_model(model, test_points):
    """Test the trained model on specific points"""
    print(f"\n🧪 Testing model on points: {test_points}")
    
    with torch.no_grad():
        test_tensor = torch.tensor(test_points).float()
        predictions = model(test_tensor)
        
        for point, pred in zip(test_points, predictions):
            print(f"Input: {point[0]:4.1f} → Prediction: {pred.item():.4f}")

# ============================================================================
# MAIN EXPERIMENT: Comparing Same vs Random Weight Initialization
# ============================================================================

print("=" * 80)
print("🔬 EXPERIMENT: Impact of Weight Initialization")
print("=" * 80)

# Create the dataset
X, Y = create_data()

# Network architecture parameters
D_in = 1    # Input dimension
H = 2       # Hidden layer size
D_out = 1   # Output dimension
learning_rate = 0.1

print(f"\n🏗️  Network Architecture: {D_in} → {H} → {D_out}")
print(f"📚 Learning Rate: {learning_rate}")

# ============================================================================
# Experiment 1: Same Weight Initialization (Problematic)
# ============================================================================

print("\n" + "="*50)
print("🔴 EXPERIMENT 1: SAME WEIGHT INITIALIZATION")
print("="*50)

model1 = Net(D_in, H, D_out)
initialize_weights_same(model1)

optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
cost1 = train(Y, X, model1, optimizer1, criterion_cross, epochs=1000)

# Analyze final weights
model1.print_weights("Final weights after training (same initialization)")

# Test the model
test_model(model1, [[-2.0], [0.0], [2.0]])

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(cost1, 'r-', linewidth=2, label='Same Weight Init')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Training Loss: Same Weight Initialization')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ============================================================================
# Experiment 2: Random Weight Initialization (Better)
# ============================================================================

print("\n" + "="*50)
print("🟢 EXPERIMENT 2: RANDOM WEIGHT INITIALIZATION")
print("="*50)

model2 = Net(D_in, H, D_out)
initialize_weights_random(model2)

optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)
cost2 = train(Y, X, model2, optimizer2, criterion_cross, epochs=1000)

# Analyze final weights
model2.print_weights("Final weights after training (random initialization)")

# Test the model
test_model(model2, [[-2.0], [0.0], [2.0]])

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(cost2, 'g-', linewidth=2, label='Random Weight Init')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Training Loss: Random Weight Initialization')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ============================================================================
# Compare Results
# ============================================================================

print("\n" + "="*50)
print("📊 COMPARISON OF RESULTS")
print("="*50)

plt.figure(figsize=(12, 8))
plt.plot(cost1, 'r-', linewidth=2, label='Same Weight Init', alpha=0.8)
plt.plot(cost2, 'g-', linewidth=2, label='Random Weight Init', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Training Loss Comparison: Same vs Random Weight Initialization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to better see differences
plt.show()

print(f"\n📈 Final Loss Comparison:")
print(f"Same Weight Init:   {cost1[-1]:.6f}")
print(f"Random Weight Init: {cost2[-1]:.6f}")

improvement = ((cost1[-1] - cost2[-1]) / cost1[-1]) * 100
print(f"Improvement: {improvement:.2f}%")

print("\n" + "="*80)
print("🎯 KEY TAKEAWAYS:")
print("="*80)
print("1. 🔴 Same weights → Symmetric neurons → Limited learning capacity")
print("2. 🟢 Random weights → Diverse neurons → Better representation learning")
print("3. 💡 Weight initialization is crucial for breaking symmetry")
print("4. 📊 Random init typically leads to faster convergence and better results")
print("="*80)
