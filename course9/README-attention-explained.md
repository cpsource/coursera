Great question — yes, in transformer-based models like BERT or GPT, **attention is applied across all words in a sequence, but the mechanism computes attention scores for each word with respect to all others**. So it's **not "word by word" sequentially**, but rather **word-by-word in parallel**.

Let me break it down:

---

### 🔁 How Attention Works with Words

Suppose we have a sentence:

> "The cat sat on the mat"

Let’s say the embedding for each word becomes part of a matrix:

```python
# Shape: (sequence_length, embedding_dim)
X = [
  x₁,  # "The"
  x₂,  # "cat"
  x₃,  # "sat"
  ...
]
```

Then we project these into:

* Queries: $Q = XW^Q$
* Keys: $K = XW^K$
* Values: $V = XW^V$

Each word gets its own **query, key, and value vector**.

---

### 💡 Core Idea:

For every word (like "cat"), we calculate how much attention it should **pay to every other word** (including itself), based on how similar their query and key vectors are.

So:

* **"cat" attends to "The"**
* **"cat" attends to "sat"**
* ...
  All at once using matrix multiplication:

```python
scores = Q @ K.T
```

Then softmax and apply to values:

```python
attention_weights = softmax(scores)
output = attention_weights @ V
```

---

### ✅ So in summary:

| Question                               | Answer                                                                                                        |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Is attention applied "word by word"?   | Each word computes attention **with every other word** in parallel.                                           |
| Is it sequential like RNNs?            | No — it's all **parallel**, enabling much faster computation.                                                 |
| Do words attend to previous ones only? | In encoder-decoder models (like GPT), yes — with masks. In others like BERT, they attend **bidirectionally**. |

Would you like a code example showing this attention applied across a whole sentence with multiple word embeddings?


