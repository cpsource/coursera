In the context of **neural networks**, especially in **language models**, **temperature** is a parameter that controls the **randomness** of the model's output during **sampling**.

---

### 🔥 What is "Temperature"?

When a model predicts the next token, it assigns a **probability** to each token in the vocabulary. These probabilities come from a **softmax function** applied to the model’s raw scores (logits).

**Temperature (T)** modifies those logits **before** softmax like this:

$$
P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Where:

* $z_i$ is the logit for token $i$
* $T$ is the **temperature**

---

### 🌡️ How it affects output:

| Temperature | Effect                    | Behavior                                             |
| ----------- | ------------------------- | ---------------------------------------------------- |
| `T = 1.0`   | No change                 | Normal sampling                                      |
| `T < 1.0`   | Sharpens the distribution | More **deterministic**, high-prob tokens more likely |
| `T > 1.0`   | Flattens the distribution | More **random**, low-prob tokens more likely         |

---

### 📌 Example:

Suppose logits before softmax give:

| Token    | Logit | Softmax @ T=1 | Softmax @ T=0.5 | Softmax @ T=2 |
| -------- | ----- | ------------- | --------------- | ------------- |
| "Paris"  | 3.0   | 0.80          | 0.95            | 0.65          |
| "London" | 2.5   | 0.15          | 0.04            | 0.25          |
| "Berlin" | 1.5   | 0.05          | 0.01            | 0.10          |

* **T = 0.5** → response becomes **more confident**, almost always “Paris”
* **T = 2.0** → response becomes **more diverse**, more chance of “London” or “Berlin”

---

### 🤖 Used in:

* **Text generation** (chatbots, story writing)
* **Creative tasks** where balance between randomness and coherence matters

---

### Summary:

> **Temperature controls how confident or random a neural network is when sampling its output.**

* **Lower T = more focused, deterministic**
* **Higher T = more exploratory, creative**

Let me know if you'd like to try running examples with different temperature settings!


