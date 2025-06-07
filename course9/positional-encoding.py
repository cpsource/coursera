# positional encoding example

import numpy as np
import torch
import torch.nn as nn

# === Step 1: Manual Positional Encoding Function ===
def positional_encoding(pos, d_model):
    """
    Compute positional encoding for a single position using sin/cos.
    """
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        angle = pos / (10000 ** (i / d_model))
        pe[i] = np.sin(angle)
        if i + 1 < d_model:
            pe[i + 1] = np.cos(angle)
    return pe

# === Step 2: Show encodings for first few positions ===
print("\nManual Positional Encodings for d_model = 4:\n")
for pos in range(5):
    encoded = positional_encoding(pos, 4)
    print(f"Position {pos}: {encoded}")

# === Step 3: PositionalEncoding Class in PyTorch ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        print("\nInput to PositionalEncoding (x):")
        print(x)
        print("\nAdded Positional Encoding:")
        out = x + self.pe[:, :x.size(1)]
        print(out)
        return out

# === Step 4: Simulate word embeddings and apply Positional Encoding ===
# Example: 3-token sentence, each with 4-dimensional embedding
embedding_dim = 4
seq_len = 3
batch_size = 1

# Simulate input embeddings (random for demo)
x = torch.randn(batch_size, seq_len, embedding_dim)

# Create PositionalEncoding layer
pos_encoder = PositionalEncoding(d_model=embedding_dim, max_len=50)

# Apply positional encoding to embeddings
pos_encoded_output = pos_encoder(x)

# === Step 5: Final Output ===
print("\nFinal result (shape):", pos_encoded_output.shape)

