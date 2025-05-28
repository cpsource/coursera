# Neural Networks More Hidden Neurons - FIXED VERSION

import torch
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def get_hist(model, data_set):
    """Fixed histogram function to visualize activations"""
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        x = model.linear1(data_set.x)  # Get pre-activation values
        activations_layer1 = torch.sigmoid(x)  # Apply sigmoid
        
        # Plot histogram of first layer activations
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(x.numpy().flatten(), bins=20, density=True, alpha=0.7, color='blue')
        plt.title("Layer 1 Pre-Activation (Linear Output)")
        plt.xlabel("Value")
        plt.ylabel("Density")
        
        plt.subplot(1, 2, 2)
        plt.hist(activations_layer1.numpy().flatten(), bins=20, density=True, alpha=0.7, color='red')
        plt.title("Layer 1 Post-Activation (Sigmoid Output)")
        plt.xlabel("Activation")
        plt.ylabel("Density")
        
        plt.tight_layout()
        plt.show()

def PlotStuff(X, Y, model=None, leg=True):
    """Fixed plotting function"""
    plt.figure(figsize=(10, 6))
    
    # Plot training points
    plt.plot(X[Y.flatten()==0].numpy(), Y[Y.flatten()==0].numpy(), 'or', 
             label='training points y=0', markersize=4)
    plt.plot(X[Y.flatten()==1].numpy(), Y[Y.flatten()==1].numpy(), 'ob', 
             label='training points y=1', markersize=4)

    if model is not None:
        model.eval()
        with torch.no_grad():
            predictions = model(X)
            plt.plot(X.numpy(), predictions.detach().numpy(), 'g-', 
                    label='neural network', linewidth=2)

    if leg:
        plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()

class Data(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20, 20, 100).view(-1, 1)
        
        # Create target pattern: 1 for x in (-10,-5) and (5,10), 0 elsewhere
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:,0] > -10) & (self.x[:,0] < -5)] = 1
        self.y[(self.x[:,0] > 5) & (self.x[:,0] < 10)] = 1
        self.y = self.y.view(-1, 1)
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

# OPTION 1: Fixed version of original Net class
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# OPTION 2: More stable version using BCEWithLogitsLoss
class NetWithLogits(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NetWithLogits, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)  # No sigmoid here - raw logits
        return x
    
    def predict_proba(self, x):
        """Get probabilities for visualization"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

def train(data_set, model, criterion, train_loader, optimizer, epochs=5, plot_number=10):
    """Fixed training function"""
    cost = []
    model.train()  # Set to training mode

    for epoch in range(epochs):
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        if epoch % plot_number == 0:
            print(f"Epoch {epoch}, Loss: {total:.4f}")
            PlotStuff(data_set.x, data_set.y, model)

        cost.append(total)
    
    # Plot final cost curve
    plt.figure()
    plt.plot(cost)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return cost

# Create dataset and visualize
data_set = Data()
print("Dataset created with pattern: y=1 for x in (-10,-5) and (5,10)")
PlotStuff(data_set.x, data_set.y, leg=True)

# TRAINING WITH ORIGINAL APPROACH (with fixes)
print("\n" + "="*50)
print("TRAINING MODEL 1: Original approach with BCE Loss")
print("="*50)

torch.manual_seed(0)
model1 = Net(1, 9, 1)
learning_rate = 0.1
criterion1 = nn.BCELoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
train_loader = DataLoader(dataset=data_set, batch_size=100)

COST1 = train(data_set, model1, criterion1, train_loader, optimizer1, epochs=600, plot_number=200)

# Show final results
print("Final results for Model 1:")
PlotStuff(data_set.x, data_set.y, model1)

# Visualize activations
print("Activation histograms for Model 1:")
get_hist(model1, data_set)

# TRAINING WITH IMPROVED APPROACH
print("\n" + "="*50)
print("TRAINING MODEL 2: Improved approach with BCEWithLogitsLoss")
print("="*50)

torch.manual_seed(0)
model2 = NetWithLogits(1, 9, 1)
criterion2 = nn.BCEWithLogitsLoss()  # More numerically stable
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

# Custom training loop for model with logits
def train_with_logits(data_set, model, criterion, train_loader, optimizer, epochs=5, plot_number=10):
    cost = []
    model.train()

    for epoch in range(epochs):
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        if epoch % plot_number == 0:
            print(f"Epoch {epoch}, Loss: {total:.4f}")
            # For plotting, we need probabilities
            model.eval()
            with torch.no_grad():
                probs = model.predict_proba(data_set.x)
                plt.figure(figsize=(10, 6))
                plt.plot(data_set.x[data_set.y.flatten()==0].numpy(), 
                        data_set.y[data_set.y.flatten()==0].numpy(), 'or', 
                        label='y=0', markersize=4)
                plt.plot(data_set.x[data_set.y.flatten()==1].numpy(), 
                        data_set.y[data_set.y.flatten()==1].numpy(), 'ob', 
                        label='y=1', markersize=4)
                plt.plot(data_set.x.numpy(), probs.numpy(), 'g-', 
                        label='neural network', linewidth=2)
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Model 2 - Epoch {epoch}')
                plt.grid(True, alpha=0.3)
                plt.show()
            model.train()

        cost.append(total)
    
    plt.figure()
    plt.plot(cost)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Over Time (Model 2)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return cost

COST2 = train_with_logits(data_set, model2, criterion2, train_loader, optimizer2, epochs=600, plot_number=200)

# SEQUENTIAL MODEL EXAMPLE (Fixed)
print("\n" + "="*50)
print("TRAINING MODEL 3: Sequential model")
print("="*50)

torch.manual_seed(0)
model3 = torch.nn.Sequential(
    torch.nn.Linear(1, 6),
    torch.nn.Sigmoid(),
    torch.nn.Linear(6, 1),
    torch.nn.Sigmoid()
)

criterion3 = nn.BCELoss()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)

COST3 = train(data_set, model3, criterion3, train_loader, optimizer3, epochs=600, plot_number=200)

# COMPARE ALL MODELS
print("\n" + "="*50)
print("COMPARING ALL MODELS")
print("="*50)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(COST1)
plt.title('Model 1: Net class (9 hidden)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(COST2)
plt.title('Model 2: With Logits (9 hidden)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(COST3)
plt.title('Model 3: Sequential (6 hidden)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Test all models on the same data
print("\nFinal comparison on test points:")
test_points = torch.tensor([[-12], [-7.5], [0], [7.5], [12]]).float()
print("Test points:", test_points.flatten().tolist())

with torch.no_grad():
    pred1 = model1(test_points)
    pred2 = model2.predict_proba(test_points)
    pred3 = model3(test_points)
    
    print("Model 1 predictions:", [f"{p.item():.3f}" for p in pred1])
    print("Model 2 predictions:", [f"{p.item():.3f}" for p in pred2])
    print("Model 3 predictions:", [f"{p.item():.3f}" for p in pred3])

print("\nExpected: [0, 1, 0, 1, 0] (approximately)")
