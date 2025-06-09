import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * i) / d_model)
            pe[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(angle)
    return pe

def main():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    embed_dim = 16

    pos_encoding = get_positional_encoding(seq_len, embed_dim)

    print("Tokens:", tokens)
    print("Positional Encoding Matrix (shape = {}):".format(pos_encoding.shape))
    print(pos_encoding)

    plt.figure(figsize=(10, 6))
    for i in range(4):
        plt.plot(range(seq_len), pos_encoding[:, i], label=f"dim {i}")
    plt.title("Sinusoidal Positional Encoding (First 4 Dimensions)")
    plt.xlabel("Token Position")
    plt.ylabel("Encoding Value")
    plt.xticks(range(seq_len), tokens)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
