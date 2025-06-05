import torch
import torch.nn as nn

# 1. Define the sentence
sentence = "the quick brown fox jumped over the lazy dog"
words = sentence.split()
print("Original sentence:", sentence)
print("Words:", words)

# 2. Build a vocabulary from unique words
vocab = {word: idx for idx, word in enumerate(sorted(set(words)))}
print("\nVocabulary:")
for word, idx in vocab.items():
    print(f"  {word} → {idx}")

# 3. Convert words to indices
token_indices = [vocab[word] for word in words]
print("\nToken indices for the sentence:")
print(token_indices)

# 4. Simulate a single bag of words (1 sentence)
input_tensor = torch.tensor(token_indices, dtype=torch.long)
offsets_tensor = torch.tensor([0], dtype=torch.long)  # single bag starts at position 0

print("\nInput tensor (flattened):", input_tensor)
print("Offsets tensor:", offsets_tensor)

# 5. Create the embedding bag layer
vocab_size = len(vocab)
embedding_dim = 5  # Keep small for printing
embedding_bag = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embedding_dim, mode='mean')

# 6. Forward pass: get the average embedding for the sentence
output = embedding_bag(input_tensor, offsets_tensor)

print("\n--- EmbeddingBag Output ---")
print("Shape:", output.shape)
print("Embedding (averaged vector):")
print(output)

# Optional: Print raw embeddings for each word
print("\nRaw embeddings for each word:")
for word in words:
    idx = vocab[word]
    vector = embedding_bag.weight[idx]
    print(f"{word:<6} → {vector.tolist()}")

