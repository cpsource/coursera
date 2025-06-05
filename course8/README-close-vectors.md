Great question! To compare two vectors and **see if they are “close”**, you typically measure the **similarity** or **distance** between them using **vector math**.

---

## ✅ Common Ways to Compare Vectors

| Method                 | Meaning                                     | Range   | Closer =                         |
| ---------------------- | ------------------------------------------- | ------- | -------------------------------- |
| **Cosine similarity**  | Measures angle between vectors              | -1 to 1 | Closer to 1                      |
| **Euclidean distance** | Measures straight-line distance in space    | 0 to ∞  | Closer to 0                      |
| **Dot product**        | Measures overlap in direction and magnitude | -∞ to ∞ | Higher is closer (if normalized) |

---

### 🧠 1. **Cosine Similarity** (most common in NLP)

$$
\text{cos\_sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

* Measures how aligned two vectors are
* **Ignores magnitude**, focuses on direction

#### 📦 In PyTorch:

```python
import torch
import torch.nn.functional as F

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.1, 2.1, 2.9])

cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
print("Cosine similarity:", cos_sim.item())
```

---

### 🧠 2. **Euclidean Distance**

$$
\text{distance}(A, B) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \dots}
$$

#### 📦 In PyTorch:

```python
euclidean = torch.dist(a, b, p=2)
print("Euclidean distance:", euclidean.item())
```

---

### ✅ When to Use What?

| Use Case                        | Recommended           |
| ------------------------------- | --------------------- |
| Word embeddings / meaning       | Cosine similarity 👍  |
| Physical vectors / spatial data | Euclidean distance    |
| Already normalized vectors      | Dot product works too |

---

### 🔍 Interpretation

If you get:

* `cosine_similarity ≈ 1.0` → Very close / semantically similar
* `cosine_similarity ≈ 0` → Unrelated
* `euclidean_distance ≈ 0` → Identical vectors

---

Let me know if you want to compare real embeddings like `"king"` vs `"queen"` or `"cat"` vs `"dog"` using Word2Vec or BERT!

