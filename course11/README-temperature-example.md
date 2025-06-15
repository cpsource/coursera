Let’s walk through the **detailed math for temperature = 1** in your softmax example.

---

### 🔢 Input logits:

| Token  | Logit |
| ------ | ----- |
| Paris  | 3.0   |
| London | 2.5   |
| Berlin | 1.5   |

We're using **temperature $T = 1$**, so the softmax is:

$$
P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

### ✅ Step 1: Compute exponentials of the logits

$$
\begin{aligned}
e^{3.0} &= 20.0855 \\
e^{2.5} &= 12.1825 \\
e^{1.5} &= 4.4817 \\
\end{aligned}
$$

---

### ✅ Step 2: Sum of exponentials

$$
\text{Sum} = 20.0855 + 12.1825 + 4.4817 = 36.7497
$$

---

### ✅ Step 3: Compute softmax probabilities

$$
\begin{aligned}
P(\text{Paris}) &= \frac{20.0855}{36.7497} ≈ 0.5464 \\
P(\text{London}) &= \frac{12.1825}{36.7497} ≈ 0.3315 \\
P(\text{Berlin}) &= \frac{4.4817}{36.7497} ≈ 0.1220 \\
\end{aligned}
$$

---

### 📊 Final Output (Softmax @ T = 1):

| Token  | Probability |
| ------ | ----------- |
| Paris  | **0.5464**  |
| London | **0.3315**  |
| Berlin | **0.1220**  |

---

Let me know if you'd like to walk through **T = 0.5** or **T = 2.0** next.


