import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define inputs (3 samples, 2 features each)
X = torch.tensor([[1.0, 2.0],
                  [2.0, 1.0],
                  [0.5, 0.5]])

# Binary target labels
y = torch.tensor([[1.0], [0.0], [1.0]])

# Define logistic regression model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

# Binary cross-entropy loss
criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Store losses
losses = []

# Train for 1000 epochs
for epoch in range(1000):
    optimizer.zero_grad()
    yhat = model(X)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Plot loss
plt.plot(range(1, 1001), losses, marker='o')
plt.title("Binary Cross-Entropy Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

