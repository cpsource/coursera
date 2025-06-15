import numpy as np
import random

# Generate 10 logits evenly spaced between 1 and 10
logits = np.linspace(1, 10, 25)

# Define softmax with temperature
def softmax_with_temperature(logits, T):
    z = logits / T
    e_z = np.exp(z - np.max(z))  # numerical stability
    return e_z / e_z.sum()

# Display softmax probabilities at various temperatures
temperatures = np.arange(0.5, 1.6, 0.1)
print("Softmax outputs at various temperatures:")
for T in temperatures:
    probs = softmax_with_temperature(logits, T)
    print(f"{probs} {round(T, 1)}")

# --------- TOP-K SELECTION --------------
# Set a specific temperature for final display

for final_T in temperatures:
    probs_final = softmax_with_temperature(logits, final_T)

    # Choose top-k
    k = 3

    # argsort - sort in ascending order
    # [-k] - slices the last k elements from the array
    # [::-1] - reverse array - [start:stop:step] - since step is minus, reverse order
    
    top_k_indices = np.argsort(probs_final)[-k:][::-1]
    top_k_probs = probs_final[top_k_indices]
    top_k_logits = logits[top_k_indices]

    # Display
    print(f"\nTop-{k} results at T = {round(final_T,1)}:")
    top_k_indices = np.argsort(probs_final)[-k:][::-1]
    for i in range(k):
        print(f"Logit: {top_k_logits[i]:.1f}, Probability: {top_k_probs[i]:.4f}")

    # Random selection from Top-K
    selected_index = random.choice(top_k_indices)
    selected_logit = logits[selected_index]
    selected_prob = probs_final[selected_index]        

    print(f"\n🎯 Randomly selected from Top-{k}:")
    print(f"Logit: {selected_logit:.1f}, Probability: {selected_prob:.4f}")

# --- TOP-P (Nucleus Sampling) ---
print(f"\nTop-p results at T = {final_T}:")
sorted_indices = np.argsort(probs_final)[::-1]  # descending order
sorted_probs = probs_final[sorted_indices]
sorted_logits = logits[sorted_indices]

for p in np.arange(0.1, 1.0, 0.1):
    cumulative = 0.0
    selected_indices = []
    
    for i, prob in enumerate(sorted_probs):
        cumulative += prob
        selected_indices.append(i)
        if cumulative >= p:
            break
    
    print(f"\nTop-p = {round(p,1)} cumulative probability ≥ {p}")
    for idx in selected_indices:
        print(f"  Logit: {sorted_logits[idx]:.1f}, Probability: {sorted_probs[idx]:.4f}")
