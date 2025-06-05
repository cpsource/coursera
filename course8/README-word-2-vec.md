### 🧠 What Is **Word2Vec**?

**Word2Vec** is a technique developed by Google (Mikolov et al., 2013) that turns **words into dense vector representations** (also called **word embeddings**) that **capture meaning and relationships** between words.

---

## ✅ Key Idea:

Instead of representing words as **one-hot vectors** (which are sparse and meaningless), Word2Vec learns **dense vectors** where:

* **Similar words** have **similar vectors**
* The **geometry** (angles/distance) between word vectors reflects **semantic relationships**

---

## ⚙️ How It Works

Word2Vec comes in **two main variants**:

| Variant       | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| **CBOW**      | Predict the **target word** from surrounding **context words** |
| **Skip-Gram** | Predict **context words** given the **target word**            |

Both use a **shallow neural network** to learn embeddings.

---

### 🔁 Example:

If you train on the sentence:

> "I like deep learning and neural networks"

Word2Vec might learn:

* `"deep"` and `"learning"` have similar vectors
* Vector(`king`) - Vector(`man`) + Vector(`woman`) ≈ Vector(`queen`) 🧠

---

## 📐 Output

* A **word embedding matrix** where:

  * Each row is a word
  * Each column is a learned feature
  * Common sizes: 100–300 dimensions

---

## 🧠 Why It Matters

| Benefit               | Description                               |
| --------------------- | ----------------------------------------- |
| 🚀 Efficient          | Trains quickly on large corpora           |
| 📚 Semantically aware | Captures analogies, similarities          |
| 🔄 Usable everywhere  | Used in classification, translation, etc. |

---

## 🧪 Use in Code (via Gensim)

```python
from gensim.models import Word2Vec

sentences = [["i", "like", "dogs"], ["i", "like", "cats"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

print(model.wv["dogs"])  # 100-dimensional vector
```

---

Let me know if you'd like to train your own Word2Vec model or visualize word relationships in 2D!


