### 🔒 What is **Masked Multi-Head Attention** in Neural Networks?

**Masked Multi-Head Attention** is a key component in **transformer-based models** like GPT, and is especially used in **autoregressive language models** (i.e., models that predict the next word/token one at a time).

---

### ✅ Definition:

> **Masked multi-head attention** is a mechanism that prevents each position in a sequence from "seeing" or attending to future positions when computing attention scores.

---

### 🧠 Why Do We Mask?

When generating text one word at a time (autoregression), the model **must not peek at future words**.
We use **masking** to enforce this.

Example:
If the sequence is:
`["I", "love", "to", "eat", "pizza"]`
→ while generating "eat", the model **must not see** "pizza".

---

### 🧱 How It Works:

1. **Multi-head attention** splits the input into multiple heads.
2. Each head performs **scaled dot-product attention**:

   $$
   \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + \text{mask} \right) V
   $$
3. The **mask** is typically a **triangular matrix** that sets future positions to `-∞` (before softmax), making them zero probability.

---

### 🔍 Visual (Causal Masking):

| t | 0 | 1 | 2 | 3 |
| - | - | - | - | - |
| 0 | ✅ | ❌ | ❌ | ❌ |
| 1 | ✅ | ✅ | ❌ | ❌ |
| 2 | ✅ | ✅ | ✅ | ❌ |
| 3 | ✅ | ✅ | ✅ | ✅ |

Each token can attend only to **current and previous** positions.

---

### 🔄 Multi-Head Aspect:

* Multiple attention heads allow the model to **learn different types of dependencies** (syntax, semantics, etc.) in parallel.
* Each head applies masking **independently**, then their outputs are concatenated and projected.

---

### 🔑 Use Case:

**GPT models, decoder blocks in transformers**, and any **language model with next-token prediction** use masked multi-head attention to enforce causality.

---

Would you like a code snippet in PyTorch or a diagram that shows how the mask is applied during attention computation?


