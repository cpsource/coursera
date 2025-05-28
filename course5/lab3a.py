# Simple One Hidden Layer Neural Network - FIXED VERSION

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

# The function for plotting the model
def PlotStuff(X, Y, model, epoch, leg=True):
    with torch.no_grad():  # Don't track gradients during plotting
        # Get probabilities for plotting
        logits = model(X)
        probs = torch.sigmoid(logits)
        plt.plot(X.numpy(), probs.numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')
    if leg == True:
        plt.legend()

# Define the class Net - FIXED VERSION
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        # Store intermediate values for visualization
        self.a1 = None
        self.l1 = None
        self.l2 = None
    
    def forward(self, x):
        self.l1 = self.linear1(x)
        self.a1 = torch.sigmoid(self.l1)
        self.l2 = self.linear2(self.a1)
        return self.l2  # Return raw logits (no sigmoid here!)

# Define the training function - FIXED VERSION
def train(Y, X, model, optimizer, criterion, epochs=1000):
    cost = []
    for epoch in range(epochs):
        total = 0
        for y, x in zip(Y, X):
            # Forward pass
            yhat = model(x)
            loss = criterion(yhat, y.unsqueeze(0))  # Ensure correct dimensions
            
            # Backward pass
            optimizer.zero_grad()  # Zero gradients BEFORE backward
            loss.backward()
            optimizer.step()
            
            total += loss.item()
        
        cost.append(total)
        
        if epoch % 300 == 0:    
            PlotStuff(X, Y, model, epoch, leg=True)
            plt.show()
            
            # Visualize activations
            with torch.no_grad():
                model(X)  # Forward pass to populate self.a1
                if model.a1.shape[1] >= 2:  # Only if we have at least 2 hidden units
                    plt.scatter(model.a1.numpy()[:, 0], model.a1.numpy()[:, 1], c=Y.numpy())
                    plt.title('activations')
                    plt.show()
    
    return cost

# Make some data
X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0

# OPTION 1: Fixed custom BCE loss with epsilon clamping
def criterion_cross_fixed(outputs, labels):
    eps = 1e-7
    probs = torch.sigmoid(outputs)  # Apply sigmoid to logits first
    probs = torch.clamp(probs, eps, 1 - eps)  # Prevent log(0)
    out = -1 * torch.mean(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
    return out

# OPTION 2: Use PyTorch's built-in BCE with logits (RECOMMENDED)
criterion_cross = nn.BCEWithLogitsLoss()

# Train the model
D_in = 1
H = 2
D_out = 1
learning_rate = 0.1

# Create and train model
model = Net(D_in, H, D_out)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
cost_cross = train(Y, X, model, optimizer, criterion_cross, epochs=1000)

# Plot the loss
plt.plot(cost_cross)
plt.xlabel('epoch')
plt.title('BCE loss')
plt.show()

# Make predictions - FIXED VERSION
def make_predictions(model, X_test):
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        return probs, predictions

# Test predictions
x_test = torch.tensor([[0.0]])
probs, preds = make_predictions(model, x_test)
print(f"Input: {x_test.item():.1f}, Probability: {probs.item():.4f}, Prediction: {preds.item()}")

X_test = torch.tensor([[0.0], [2.0], [3.0]])
probs, preds = make_predictions(model, X_test)
print(f"Test inputs: {X_test.flatten().tolist()}")
print(f"Probabilities: {probs.flatten().tolist()}")
print(f"Predictions: {preds.flatten().tolist()}")

# Compare with MSE Loss
print("\n" + "="*50)
print("Training with MSE Loss for comparison:")

model_mse = Net(D_in, H, D_out)
criterion_mse = nn.MSELoss()
optimizer_mse = torch.optim.SGD(model_mse.parameters(), lr=learning_rate)

# For MSE, we need sigmoid in forward pass since MSE expects probabilities
class NetForMSE(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NetForMSE, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.a1 = None
        self.l1 = None
        self.l2 = None
    
    def forward(self, x):
        self.l1 = self.linear1(x)
        self.a1 = torch.sigmoid(self.l1)
        self.l2 = self.linear2(self.a1)
        return torch.sigmoid(self.l2)  # Apply sigmoid for MSE

model_mse = NetForMSE(D_in, H, D_out)
optimizer_mse = torch.optim.SGD(model_mse.parameters(), lr=learning_rate)
cost_mse = train(Y, X, model_mse, optimizer_mse, criterion_mse, epochs=1000)

plt.plot(cost_mse)
plt.xlabel('epoch')
plt.title('MSE loss')
plt.show()

# Compare losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cost_cross)
plt.title('BCE Loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(cost_mse)
plt.title('MSE Loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()
