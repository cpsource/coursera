### 🧠 What Is **Positional Encoding**?

**Positional encoding** is a technique used in **Transformer models** (like BERT and GPT) to inject information about the **position/order of tokens** in a sequence — because, unlike RNNs, **Transformers have no built-in notion of order**.

---

### ✅ Why Is It Needed?

* Transformers treat all tokens **in parallel**, not sequentially.
* Without positional information, the model wouldn’t know if a word came **before** or **after** another.

---

## 🔣 How It Works

Positional encodings are **vectors** that are **added to the input embeddings**. They encode each token's position using **sine and cosine functions** of different frequencies.

---

### 📐 Formula (from "Attention Is All You Need"):

For a token at position $pos$ and dimension $i$:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Where:

* $d$ is the embedding dimension
* Even indices use `sin`, odd indices use `cos`

---

## 🧪 Simple Example (Manually)

Assume:

* Sequence: `["I", "like", "cats"]`
* Embedding dim $d = 4$

Compute the **positional encoding** for positions `pos = 0, 1, 2`:

```python
import numpy as np

def positional_encoding(pos, d_model):
    PE = np.zeros(d_model)
    for i in range(0, d_model, 2):
        angle = pos / (10000 ** (i / d_model))
        PE[i] = np.sin(angle)
        if i + 1 < d_model:
            PE[i+1] = np.cos(angle)
    return PE

for pos in range(3):
    print(f"Position {pos}: {positional_encoding(pos, 4)}")
```

**Output (approx.):**

```
Position 0: [0.         1.         0.         1.        ]
Position 1: [0.0001     0.9999     0.00000316  1.        ]
Position 2: [0.0002     0.9998     0.00000632  1.        ]
```

---

### 📦 In Practice (PyTorch):

```python
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## ✅ Summary

| Feature  | Explanation                                                 |
| -------- | ----------------------------------------------------------- |
| Purpose  | Inject order info into non-sequential models (Transformers) |
| Based on | Sine/cosine functions of position + dimension               |
| Used in  | BERT, GPT, T5, ViT, etc.                                    |
| Adds to  | Input embeddings before feeding into Transformer layers     |

---

Let me know if you'd like a visual of how sine/cosine patterns look across positions!


