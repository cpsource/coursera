Great question!

The answer depends on the **type of positional encoding** you're using in your Transformer model:

---

## ✅ 1. **Sinusoidal Positional Encoding** (used in original Transformer paper)

**🔹 Not trainable**

* These are **fixed** values generated using sine and cosine functions.
* The model **does not learn** them — they are **precomputed** and added to the input embeddings.

### Example (PyTorch-style):

```python
position_encodings = get_sin_cos(pos, d_model)  # Precomputed
x = x + position_encodings  # No parameters to learn
```

> ✅ Simple, fast
> ❌ Not flexible or adaptive to your specific data

---

## ✅ 2. **Learned Positional Embeddings** (used in BERT, GPT, etc.)

**🔹 Trainable**

* These are **learnable embedding vectors**, just like word embeddings.
* One vector per position (e.g., for position 0, 1, 2, ..., 512)

### Example:

```python
self.positional_embeddings = nn.Embedding(max_seq_len, embedding_dim)
```

During training:

* These **weights (and biases, if applicable)** are updated using **backpropagation**
* The model learns **how important each position is** based on the task

> ✅ Adaptive to your data
> ❌ May not generalize well to longer unseen sequences

---

## ✅ Summary Table

| Type                          | Trainable? | Used By                    |
| ----------------------------- | ---------- | -------------------------- |
| Sinusoidal (fixed)            | ❌ No       | Transformer (original), T5 |
| Learned positional embeddings | ✅ Yes      | BERT, GPT, RoBERTa         |

---

## 🧠 Bonus: Biases?

* **Positional encodings** themselves typically do **not use biases**
* If you're using an `nn.Linear` layer after adding them, **that layer** may have biases — but not the encoding itself

---

Let me know if you want to compare both styles in code or visualize how the model learns positional weights!


