import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 100)

# Different activation functions
sigmoid_y = torch.sigmoid(x)
tanh_y = torch.tanh(x)
relu_y = torch.relu(x)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid_y)
plt.title('Sigmoid: Saturates (bad gradients)')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, tanh_y)
plt.title('Tanh: Better than sigmoid, still saturates')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, relu_y)
plt.title('ReLU: Simple, no saturation')
plt.grid(True)

plt.tight_layout()
plt.show()
