### 🧠 What is `nn.EmbeddingBag` in PyTorch?

`nn.EmbeddingBag` is a **compressed embedding layer** in PyTorch designed to handle **bags of tokens** (e.g., word indices) and **efficiently compute aggregated embeddings** — like **sums** or **averages** — in a single operation.

---

## ✅ Key Features:

| Feature                         | Description                                                     |
| ------------------------------- | --------------------------------------------------------------- |
| Combines lookup + reduction     | Computes **embedding + aggregation (sum/mean/max)** in one step |
| Efficient for BOW/N-gram models | Especially useful for **bag-of-words** or **n-gram** models     |
| No intermediate tensor          | Saves memory and computation                                    |

---

### 🔧 Basic Syntax:

```python
import torch.nn as nn

embedding_bag = nn.EmbeddingBag(num_embeddings=10, embedding_dim=5, mode='mean')
```

* `num_embeddings`: Vocabulary size
* `embedding_dim`: Dimension of each embedding vector
* `mode`: Aggregation mode — `"mean"`, `"sum"`, or `"max"`

---

## 🔍 How It Works

It expects:

* **`input`**: Flattened list of token indices
* **`offsets`**: Starting index of each "bag" in the `input`

```python
import torch

input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])     # token indices (all in one list)
offsets = torch.tensor([0, 4])                    # first bag = tokens[0:4], second = tokens[4:8]

output = embedding_bag(input, offsets)
```

So this computes:

* First row: `mean(embeddings[1], embeddings[2], embeddings[4], embeddings[5])`
* Second row: `mean(embeddings[4], embeddings[3], embeddings[2], embeddings[9])`

Result: a 2×5 tensor (2 bags, 5-dim embeddings)

---

## 🧠 Why Use `EmbeddingBag`?

* ✅ **Faster and more memory-efficient** than `nn.Embedding + torch.sum/mean`
* ✅ Used in **text classification** models like the one in PyTorch’s [Text Classification Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

---

## 📦 Example Use Case: Text Classification

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
```

---

Let me know if you'd like a visual diagram of how `offsets` work or want to build a mini classifier with it!


