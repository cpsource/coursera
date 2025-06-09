import torch
import torch.nn as nn
import numpy

# Define a small vocabulary and embedding dimension
vocab = {"I": 0, "love": 1, "deep": 2, "learning": 3}
print(f"vocab = {vocab}")

vocab_size = len(vocab)
embedding_dim = 8

# Sample tokenized sentence
tokens = ["I", "love", "deep", "learning"]
print("tokens:", tokens)

token_ids = [vocab[token] for token in tokens]
print("Token IDs:", token_ids)

# Note: the unsqueeze adds a batch dimension to the front of token_ids
#  - a shape needed for further processing
token_tensor = torch.tensor(token_ids).unsqueeze(0)  # Shape: (1, 4)
print(f"token_tensor = {token_tensor}")

# Define the embedding layer
#  vocab_size = len(vocab)
#  embedding_dim = 8
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
print("Embedding weights as numpy:\n", embedding_layer.weight.detach().numpy())
#
# Note, there's a lookup table under the hood that gets trained by back-prop
# for now though, it's initialized by random numbers
#
#print(f"embedding_layer = {embedding_layer}")

# Convert tokens to embeddings
print("# Convert tokens to embeddings")
print(f"token_tensor = {token_tensor}")
sequence_tensor = embedding_layer(token_tensor)
print("Sequence tensor:\n", sequence_tensor)
print("Shape of sequence tensor:", sequence_tensor.shape)
