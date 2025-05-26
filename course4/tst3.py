import torch
import torch.nn as nn

# Input features for 3 samples (each has 2 features)
X = torch.tensor([[1.0, 2.0],
                  [2.0, 1.0],
                  [0.5, 0.5]])

# Corresponding binary labels (ground truth)
y = torch.tensor([[1.0], [0.0], [1.0]])

# Simple logistic regression model (2 → 1)
model = nn.Sequential(
    nn.Linear(2, 1),      # Linear transformation: θᵀx + b
    nn.Sigmoid()          # Apply sigmoid to get probabilities
)

# Binary Cross Entropy loss function
criterion = nn.BCELoss()

# Forward pass: predict probabilities
y_pred = model(X)

# Compute loss (uses ln inside)
loss = criterion(y_pred, y)

print("Predicted probabilities:\n", y_pred)
print("Binary cross-entropy loss:", loss.item())

