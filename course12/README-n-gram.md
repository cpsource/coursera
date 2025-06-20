An **n-gram** in neural networks (and NLP in general) refers to a **sequence of `n` consecutive tokens (words, characters, or subwords)** from a given text.

---

### 🔹 Examples:

For the sentence:

> "I love AI"

* **1-gram (unigram)**: `["I", "love", "AI"]`
* **2-gram (bigram)**: `["I love", "love AI"]`
* **3-gram (trigram)**: `["I love AI"]`

---

### 🔹 In Neural Networks:

N-grams are used to capture **local context** and **word co-occurrence**. For example:

* In **classic NLP models** (like Naive Bayes or Logistic Regression), n-grams were used as features.
* In **neural networks**, especially early ones like RNNs or CNNs for text:

  * N-grams help the model understand **word sequences** or **phrases**, not just isolated words.
  * CNNs may simulate n-gram patterns via **convolution filters** of size `n`.

---

### 🔹 In Transformers:

Modern architectures like **BERT**, **GPT**, etc., don't use explicit n-grams — they instead:

* Use **attention mechanisms** to learn context across the entire sequence.
* But still rely on **tokenization**, often at the **subword level** (e.g., `WordPiece`, `Byte Pair Encoding`), which implicitly forms variable-length "n-grams" in how the model encodes relationships.

---

### Summary:

| Term         | Meaning                                                     |
| ------------ | ----------------------------------------------------------- |
| N-gram       | Sequence of `n` tokens                                      |
| Purpose      | Capture local context and sequence dependencies             |
| Used in      | Classic models, CNNs, preprocessing, tokenizers             |
| In modern NN | Replaced by learned contextual embeddings (e.g., attention) |

Let me know if you want code examples or n-gram generation in Python.

