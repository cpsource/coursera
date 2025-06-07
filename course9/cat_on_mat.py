import numpy as np

# Step 1 - Word Mappings and Embeddings

# Mapping words to unique IDs
vocab = {
    "The": 1,
    "cat": 2,
    "sat": 3,
    "on": 4,
    "the": 5,
    "mat.": 6
}

# Let's assign simple embeddings (3-dim vectors)
embeddings = {
    1: np.array([1.0, 0.0, 0.0]),  # "The"
    2: np.array([0.9, 0.1, 0.0]),  # "cat"
    3: np.array([0.0, 1.0, 0.0]),  # "sat"
    4: np.array([0.0, 0.9, 0.1]),  # "on"
    5: np.array([1.0, 0.0, 0.1]),  # "the"
    6: np.array([0.1, 0.0, 1.0])   # "mat."
}
print(f"embeddings = {embeddings}")

# Convert sentence to list of embeddings
sentence = ["The", "cat", "sat", "on", "the", "mat."]
X = np.array([embeddings[vocab[word]] for word in sentence])
print(f"list of embeddings for sentence = {X}")

# Step 2 - Compute Attention (Scaled Dot-Product)

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def compute_attention_verbose(X, sentence):
    """
    X: numpy array of shape (sequence_length, embedding_dim)
    sentence: list of words corresponding to each row in X
    """
    d_k = X.shape[1]  # dimensionality of embeddings
    print(f"d_k = {d_k}")

    attention_outputs = []
    attention_weights_all = []

    for i in range(X.shape[0]):
        print(f"\n=== Word: '{sentence[i]}' ===")
        query = X[i]
        print(f"Query Vector: {query}")

        raw_scores = np.dot(X, query)
        print(f"Raw Scores (dot product): {raw_scores.round(3)}")

        scaled_scores = raw_scores / np.sqrt(d_k)
        print(f"Scaled Scores (/ sqrt({d_k})): {scaled_scores.round(3)}")

        weights = softmax(scaled_scores)
        print(f"Softmax Weights: {weights.round(3)}")

        attention_output = np.dot(weights, X)
        print(f"Attention Output (weighted sum): {attention_output.round(3)}")

        attention_outputs.append(attention_output)
        attention_weights_all.append(weights)

    return np.array(attention_outputs), np.array(attention_weights_all)

# STEP 3: Run and Explain

outputs, weights = compute_attention_verbose(X,sentence)

print("\n--- Final Attention Output Vectors ---")
for i, out in enumerate(outputs):
    print(f"{sentence[i]} => {out.round(3)}")


# EXPLANATION
#  Let's say we're at word "cat".
#
#  It computes dot products with all other words to get raw attention scores.
#
#  These scores are scaled and normalized with softmax to become probabilities.
#
#  Then it uses these weights to compute a weighted sum of all word vectors.
#
#  So "cat" might pay more attention to "The" or "sat" depending on the vector similarity.
#
#  This is what "attention" means:
#  Each word decides how much it should "look at" every other word when forming its understanding of the context.
#

