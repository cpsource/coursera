Absolutely! Here's a simple example of the **attention mechanism** in Python using NumPy (no deep learning frameworks required). This is a scaled dot-product attention, the core idea behind attention in models like Transformers.

---

### 🔍 **Simple Attention Example (Scaled Dot-Product Attention)**

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def attention(query, key, value):
    d_k = query.shape[-1]  # dimension of the key
    scores = np.matmul(query, key.T) / np.sqrt(d_k)  # Scaled dot-product
    weights = softmax(scores)  # Attention weights
    output = np.matmul(weights, value)  # Weighted sum of values
    return output, weights

# Example with dummy data
query = np.array([[1, 0, 1]])      # Shape (1, 3)
key   = np.array([[1, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0]])      # Shape (3, 3)
value = np.array([[10, 0],
                  [0, 10],
                  [5, 5]])         # Shape (3, 2)

output, weights = attention(query, key, value)

print("Attention weights:")
print(weights)
print("\nAttention output:")
print(output)
```

---

### 🧠 How It Works

1. **Query, Key, Value**: These are input vectors. In real NLP models, they come from learned projections of input embeddings.
2. **Dot Product**: We compute similarity between query and each key.
3. **Softmax**: We convert scores into probabilities (attention weights).
4. **Weighted Sum**: Each value is weighted by attention and summed to give the final result.

---

Would you like this rewritten using PyTorch or TensorFlow for use in an actual model layer?


