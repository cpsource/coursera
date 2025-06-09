In neural networks (NN), especially in the context of **natural language processing** and **sequence modeling**, **autoregression** refers to:

> 🔁 **A modeling strategy where the output at each time step is used as input for predicting the next time step.**

---

### 🧠 In Simple Terms:

Autoregression =
**“Predict the next item in a sequence based on everything you’ve generated so far.”**

---

### 📘 Example in Language Modeling:

Suppose you’re training a language model on:

> “I like to eat…”

The model predicts:

* next word: `pizza`

Then it takes “I like to eat pizza” and predicts:

* next word: `on`

Then:

* “I like to eat pizza on…” → predict: `Fridays`

This **step-by-step prediction**, where each output becomes input for the next step, is **autoregressive** behavior.

---

### 🧱 In Transformers (e.g., GPT):

* Autoregression is enforced via **causal masks**, so the model:

  * Can attend only to **past and present tokens**
  * ❌ Not to future tokens

---

### 🧠 Equation Style:

Autoregressive models predict:

$$
P(x_1, x_2, ..., x_n) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1, x_2) \cdots P(x_n|x_1, ..., x_{n-1})
$$

---

### 🔁 Contrast:

| Term                   | Description                                                                       |
| ---------------------- | --------------------------------------------------------------------------------- |
| **Autoregressive**     | Predict next token using previous ones                                            |
| **Non-autoregressive** | Predict all tokens in parallel (used for speed, but less accurate for generation) |

---

Would you like a simple code demo showing how autoregression works in PyTorch or a visual of attention masking?


