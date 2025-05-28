# Activation Functions Lab

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(2)

# Create a tensor range for plotting
z = torch.arange(-10, 10, 0.1).view(-1, 1)

# SIGMOID ACTIVATION
print("=== Sigmoid Activation Function ===")

# Method 1: Using nn.Sigmoid class
sig = nn.Sigmoid()
yhat_sig_class = sig(z)

plt.figure(figsize=(10, 6))
plt.plot(z.detach().numpy(), yhat_sig_class.detach().numpy(), 'b-', linewidth=2)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Sigmoid Activation Function: σ(z) = 1/(1 + e^(-z))')
plt.grid(True, alpha=0.3)
plt.show()

# Method 2: Using built-in torch.sigmoid function
yhat_sig_builtin = torch.sigmoid(z)
plt.figure(figsize=(10, 6))
plt.plot(z.numpy(), yhat_sig_builtin.numpy(), 'r-', linewidth=2)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Sigmoid Activation Function (Built-in torch.sigmoid)')
plt.grid(True, alpha=0.3)
plt.show()

# TANH ACTIVATION
print("=== Tanh Activation Function ===")

# Method 1: Using nn.Tanh class
tanh = nn.Tanh()
yhat_tanh_class = tanh(z)

plt.figure(figsize=(10, 6))
plt.plot(z.numpy(), yhat_tanh_class.numpy(), 'g-', linewidth=2)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Hyperbolic Tangent Activation Function: tanh(z)')
plt.grid(True, alpha=0.3)
plt.show()

# Method 2: Using built-in torch.tanh function
yhat_tanh_builtin = torch.tanh(z)
plt.figure(figsize=(10, 6))
plt.plot(z.numpy(), yhat_tanh_builtin.numpy(), 'm-', linewidth=2)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('Hyperbolic Tangent Activation Function (Built-in torch.tanh)')
plt.grid(True, alpha=0.3)
plt.show()

# RELU ACTIVATION
print("=== ReLU Activation Function ===")

# Method 1: Using nn.ReLU class
relu = nn.ReLU()
yhat_relu_class = relu(z)

plt.figure(figsize=(10, 6))
plt.plot(z.numpy(), yhat_relu_class.numpy(), 'orange', linewidth=2)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('ReLU Activation Function: max(0, z)')
plt.grid(True, alpha=0.3)
plt.show()

# Method 2: Using built-in torch.relu function
yhat_relu_builtin = torch.relu(z)
plt.figure(figsize=(10, 6))
plt.plot(z.numpy(), yhat_relu_builtin.numpy(), 'purple', linewidth=2)
plt.xlabel('Input (z)')
plt.ylabel('Output')
plt.title('ReLU Activation Function (Built-in torch.relu)')
plt.grid(True, alpha=0.3)
plt.show()

# COMPARISON PLOTS
print("=== Activation Functions Comparison ===")

# Compare activation functions in range (-2, 2)
x = torch.arange(-2, 2, 0.1).view(-1, 1)

plt.figure(figsize=(12, 8))
plt.plot(x.numpy(), torch.relu(x).numpy(), 'orange', label='ReLU', linewidth=2)
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), 'blue', label='Sigmoid', linewidth=2)
plt.plot(x.numpy(), torch.tanh(x).numpy(), 'green', label='Tanh', linewidth=2)
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.title('Comparison of Activation Functions (Range: -2 to 2)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

# Compare activation functions in range (-1, 1) - zoomed in view
x_zoom = torch.arange(-1, 1, 0.1).view(-1, 1)

plt.figure(figsize=(12, 8))
plt.plot(x_zoom.numpy(), torch.relu(x_zoom).numpy(), 'orange', label='ReLU', linewidth=2)
plt.plot(x_zoom.numpy(), torch.sigmoid(x_zoom).numpy(), 'blue', label='Sigmoid', linewidth=2)
plt.plot(x_zoom.numpy(), torch.tanh(x_zoom).numpy(), 'green', label='Tanh', linewidth=2)
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.title('Comparison of Activation Functions (Zoomed: -1 to 1)')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()

print("\nActivation Function Properties:")
print("• Sigmoid: Output range (0, 1) - good for binary classification")
print("• Tanh: Output range (-1, 1) - zero-centered, often better than sigmoid")
print("• ReLU: Output range [0, ∞) - most popular, solves vanishing gradient problem")
