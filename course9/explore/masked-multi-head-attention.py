import torch
import torch.nn.functional as F
import sys

# Set random seed for reproducibility
torch.manual_seed(0)

# Simulate input: 1 sentence, 4 tokens, each with 8-dim embeddings
batch_size = 1
seq_len = 4
embed_dim = 8
num_heads = 2
head_dim = embed_dim // num_heads

# Step 1: Input Embeddings
x = torch.randn(batch_size, seq_len, embed_dim)
print("Input Embeddings:\n", x, "\n")

# Step 2: Linear projections for Q, K, V
W_q = torch.nn.Linear(embed_dim, embed_dim)
W_k = torch.nn.Linear(embed_dim, embed_dim)
W_v = torch.nn.Linear(embed_dim, embed_dim)

Q = W_q(x)
K = W_k(x)
V = W_v(x)
print("Q:\n", Q, "\n")
print("K:\n", K, "\n")
print("V:\n", V, "\n")

# Step 3: Reshape to (batch, heads, seq_len, head_dim)
def reshape_heads(tensor):
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

Q = reshape_heads(Q)
K = reshape_heads(K)
V = reshape_heads(V)
print("Q reshaped for multi-head:\n", Q, "\n")
# Suppose:
# 
# batch_size = 1
# 
# seq_len = 4
# 
# embed_dim = 8
# 
# num_heads = 2
# → then head_dim = embed_dim // num_heads = 4
# 

# Q.shape = (1, 2, 4, 4)  # [batch, heads, tokens, head_dim]

# What this means:
# We now have 2 attention heads.
# 
# Each head gets a 4-dimensional view of each token (instead of all 8 at once).
# 
# Each head operates independently from the others.
# 

# Step 4: Scaled dot-product attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
print("Raw Attention Scores:\n", scores, "\n")

# Step 5: Causal Masking (prevent attention to future tokens)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print("Causal Mask:\n", mask, "\n")
scores = scores.masked_fill(mask, float('-inf'))
print("Masked Attention Scores:\n", scores, "\n")

# Step 6: Softmax normalization
attn_weights = F.softmax(scores, dim=-1)
print("Attention Weights from softmax: torch.nn.functional.(Masked Attention Scores)\n", attn_weights, "\n")

# attention_output = 0.6 * V("The") + 0.3 * V("cat") + 0.1 * V("sat")

#sys.exit(0)

# Step 7: Weighted sum of values
attn_output = torch.matmul(attn_weights, V)
print("Attention Output (per head):\n", attn_output, "\n")

# Step 8: Concatenate heads
attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
print("Final Attention Output:\n", attn_output)


