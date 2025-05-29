import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("BATCH NORMALIZATION with DIFFERENT ACTIVATION FUNCTIONS")
print("="*70)

# ========================
# HYPERPARAMETERS
# ========================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3  # Reduced for demonstration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

# ========================
# DATA LOADING
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# DIFFERENT CNN ARCHITECTURES
# ========================

class CNNWithReLU(nn.Module):
    """Standard CNN with ReLU (no batch norm needed)"""
    def __init__(self):
        super(CNNWithReLU, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNWithSigmoidNoBN(nn.Module):
    """CNN with Sigmoid (no batch norm) - should perform poorly"""
    def __init__(self):
        super(CNNWithSigmoidNoBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.sigmoid(self.conv1(x)))
        x = self.pool(torch.sigmoid(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNWithSigmoidBN(nn.Module):
    """CNN with Sigmoid + Batch Normalization - should perform much better!"""
    def __init__(self):
        super(CNNWithSigmoidBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch norm after conv1
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch norm after conv2
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)  # Batch norm for fully connected
        
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply batch norm BEFORE activation (common practice)
        x = self.pool(torch.sigmoid(self.bn1(self.conv1(x))))
        x = self.pool(torch.sigmoid(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.sigmoid(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNWithTanhBN(nn.Module):
    """CNN with Tanh + Batch Normalization"""
    def __init__(self):
        super(CNNWithTanhBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.tanh(self.bn1(self.conv1(x))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.tanh(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========================
# HOW BATCH NORMALIZATION HELPS
# ========================
print("\nHOW BATCH NORMALIZATION HELPS WITH SIGMOID/TANH:")
print("-" * 55)

def demonstrate_bn_effect():
    """Show how batch norm keeps inputs in the active region"""
    
    # Simulate some layer outputs (before activation) - use batch dimension > 1
    batch_size = 64
    features = 100
    raw_outputs = torch.randn(batch_size, features) * 5  # Random values with high variance
    
    # Flatten for analysis
    raw_flat = raw_outputs.flatten()
    
    # Without batch norm - many values will saturate sigmoid/tanh
    sigmoid_without_bn = torch.sigmoid(raw_flat)
    tanh_without_bn = torch.tanh(raw_flat)
    
    # With batch norm - normalize to have mean=0, std=1
    bn = nn.BatchNorm1d(features, affine=False)  # No learnable params for demo
    bn.eval()  # Set to eval mode to avoid the batch size error
    
    # Manually normalize to show the effect (alternative approach)
    normalized_outputs = (raw_outputs - raw_outputs.mean()) / raw_outputs.std()
    normalized_flat = normalized_outputs.flatten()
    
    sigmoid_with_bn = torch.sigmoid(normalized_flat)
    tanh_with_bn = torch.tanh(normalized_flat)
    
    print(f"Raw outputs stats:")
    print(f"  Mean: {raw_flat.mean():.3f}, Std: {raw_flat.std():.3f}")
    print(f"  Range: {raw_flat.min():.3f} to {raw_flat.max():.3f}")
    
    print(f"\nAfter Batch Norm:")
    print(f"  Mean: {normalized_flat.mean():.3f}, Std: {normalized_flat.std():.3f}")
    print(f"  Range: {normalized_flat.min():.3f} to {normalized_flat.max():.3f}")
    
    # Calculate how many activations are in the "active" region
    def count_active(activations, threshold=0.1):
        # Count activations that are away from saturation
        return ((activations > threshold) & (activations < (1-threshold))).float().mean()
    
    def count_active_tanh(activations, threshold=0.1):
        # For tanh, active region is away from -1 and 1
        return ((activations > -1+threshold) & (activations < 1-threshold)).float().mean()
    
    sigmoid_active_without = count_active(sigmoid_without_bn)
    sigmoid_active_with = count_active(sigmoid_with_bn)
    tanh_active_without = count_active_tanh(tanh_without_bn)
    tanh_active_with = count_active_tanh(tanh_with_bn)
    
    print(f"\nActivation Analysis:")
    print(f"Sigmoid - Active neurons without BN: {sigmoid_active_without:.1%}")
    print(f"Sigmoid - Active neurons with BN:    {sigmoid_active_with:.1%}")
    print(f"Tanh - Active neurons without BN:    {tanh_active_without:.1%}")
    print(f"Tanh - Active neurons with BN:       {tanh_active_with:.1%}")
    
    return raw_flat, normalized_flat, sigmoid_without_bn, sigmoid_with_bn, tanh_without_bn, tanh_with_bn

raw, norm, sig_no_bn, sig_bn, tanh_no_bn, tanh_bn = demonstrate_bn_effect()

# ========================
# VISUALIZATION
# ========================
print(f"\nVISUALIZING THE EFFECT:")
print("-" * 25)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Input distributions
ax1.hist(raw.numpy(), bins=50, alpha=0.7, label='Raw outputs', color='red')
ax1.hist(norm.numpy(), bins=50, alpha=0.7, label='After BatchNorm', color='blue')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency') 
ax1.set_title('Input Distribution: Before vs After BatchNorm')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Sigmoid activations
ax2.hist(sig_no_bn.numpy(), bins=50, alpha=0.7, label='Sigmoid w/o BN', color='red')
ax2.hist(sig_bn.numpy(), bins=50, alpha=0.7, label='Sigmoid w/ BN', color='blue')
ax2.set_xlabel('Activation Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Sigmoid Activations: Effect of BatchNorm')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Tanh activations  
ax3.hist(tanh_no_bn.numpy(), bins=50, alpha=0.7, label='Tanh w/o BN', color='red')
ax3.hist(tanh_bn.numpy(), bins=50, alpha=0.7, label='Tanh w/ BN', color='blue')
ax3.set_xlabel('Activation Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Tanh Activations: Effect of BatchNorm')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Gradient comparison
x_range = torch.linspace(-3, 3, 100)
sigmoid_grad = torch.sigmoid(x_range) * (1 - torch.sigmoid(x_range))
tanh_grad = 1 - torch.tanh(x_range)**2

ax4.plot(x_range, sigmoid_grad, 'b-', linewidth=2, label='Sigmoid gradient')
ax4.plot(x_range, tanh_grad, 'g-', linewidth=2, label='Tanh gradient')
ax4.axvspan(-1, 1, alpha=0.2, color='yellow', label='BatchNorm keeps inputs here')
ax4.set_xlabel('Input value')
ax4.set_ylabel('Gradient')
ax4.set_title('Why BatchNorm Helps: Keeps Inputs in High-Gradient Region')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================
# TRAINING COMPARISON
# ========================
print(f"\nTRAINING COMPARISON:")
print("-" * 20)

def train_model(model, model_name, epochs=EPOCHS):
    """Train a model and return training history"""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 200 == 0:
                print(f'  Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'  Epoch {epoch+1} Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

# Create models
models = {
    'ReLU (no BN)': CNNWithReLU(),
    'Sigmoid (no BN)': CNNWithSigmoidNoBN(), 
    'Sigmoid + BN': CNNWithSigmoidBN(),
    'Tanh + BN': CNNWithTanhBN()
}

# Train all models
results = {}
for name, model in models.items():
    losses, accuracies = train_model(model, name)
    results[name] = {'losses': losses, 'accuracies': accuracies}

# ========================
# RESULTS VISUALIZATION
# ========================
print(f"\nRESULTS COMPARISON:")
print("-" * 20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot training losses
for name, data in results.items():
    ax1.plot(data['losses'], marker='o', linewidth=2, label=name)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot training accuracies
for name, data in results.items():
    ax2.plot(data['accuracies'], marker='o', linewidth=2, label=name)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training Accuracy (%)')
ax2.set_title('Training Accuracy Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final results
print(f"\nFINAL RESULTS AFTER {EPOCHS} EPOCHS:")
print(f"{'Model':<15} {'Final Loss':<12} {'Final Accuracy':<15}")
print("-" * 45)

for name, data in results.items():
    final_loss = data['losses'][-1]
    final_acc = data['accuracies'][-1]
    print(f"{name:<15} {final_loss:<12.4f} {final_acc:<15.2f}%")

# ========================
# KEY INSIGHTS
# ========================
print(f"\n" + "="*70)
print("KEY INSIGHTS ABOUT BATCH NORMALIZATION WITH DIFFERENT ACTIVATIONS")
print("="*70)

print("""
1. HOW BATCH NORMALIZATION HELPS:
   ✓ Normalizes inputs to activation functions (mean=0, std=1)  
   ✓ Keeps inputs in the "active" region where gradients are strong
   ✓ Reduces internal covariate shift
   ✓ Acts as regularization (slight noise from batch statistics)

2. WHY IT RESCUES SIGMOID/TANH:
   ✓ Prevents inputs from falling into saturation regions
   ✓ Maintains higher gradient flow throughout training
   ✓ Makes these activations competitive with ReLU
   ✓ Allows deeper networks with sigmoid/tanh

3. BATCH NORM PLACEMENT:
   Common approaches:
   - Conv → BatchNorm → Activation → Pool
   - Conv → Activation → BatchNorm → Pool
   (First approach is more common)

4. PERFORMANCE RANKING (typical):
   1. ReLU (fast, simple, effective)
   2. Tanh + BatchNorm (zero-centered, good gradients)
   3. Sigmoid + BatchNorm (helped but still non-zero-centered)
   4. Sigmoid without BatchNorm (poor due to saturation)

5. WHEN TO USE EACH:
   - ReLU: Default choice, fastest training
   - Tanh + BN: When you need bounded, zero-centered activations
   - Sigmoid + BN: When you need outputs in [0,1] range
   - Modern alternatives: Swish, GELU (combine benefits)

6. COMPUTATIONAL COST:
   BatchNorm adds parameters and computation, but the improved 
   convergence often makes total training time shorter!
""")

print(f"\nCONCLUSION:")
print(f"Batch Normalization can indeed make sigmoid and tanh viable again!")
print(f"It's like giving these activation functions a 'second chance' by")
print(f"keeping their inputs in the sweet spot where they work well.")
