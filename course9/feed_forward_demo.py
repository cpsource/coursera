import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy input: batch of 2 sequences, each with 4 tokens, each token is a 6-dimensional vector
x = torch.randn(2, 4, 6)  # shape: (batch_size, sequence_length, model_dim)

# Define a simple feed-forward layer like in transformers
class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Create the layer
model_dim = 6 # the dimension of each token
hidden_dim = 12  # usually 2–4× the model_dim
ff_layer = FeedForward(model_dim, hidden_dim)

# Run the input through the feed-forward layer
output = ff_layer(x)

# Print shapes for clarity
print("Input shape: ", x.shape)
print("Output shape:", output.shape)

print(f"input {input}")
print(f"output {x.numpy()}")


