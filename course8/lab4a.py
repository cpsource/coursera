import torch
import torch.nn as nn

# Define fake vocab size and embedding dimension
vocab_size = 10
embedding_dim = 4

# Create an EmbeddingBag layer with mean pooling
embedding_bag = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embedding_dim, mode='mean')

# Simulate input: 2 sentences (bags of word indices)
# Bag 1: tokens [1, 2, 4]
# Bag 2: tokens [4, 3, 2, 9]
input = torch.tensor([1, 2, 4, 4, 3, 2, 9], dtype=torch.long)

# Offsets: marks the start of each "bag" in the input
offsets = torch.tensor([0, 3], dtype=torch.long)

# Forward pass through embedding bag
output = embedding_bag(input, offsets)

# Print results
print("Input indices:", input)
print("Offsets:", offsets)
print("Output embeddings (averaged per bag):")
print(output)

