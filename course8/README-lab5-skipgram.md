Let me explain Skip-gram and provide comprehensive documentation for that class:

## What is a Skip-gram Model?

A Skip-gram model is like a "word prophet" - given a single word, it tries to predict what words are likely to appear nearby. Think of it as the opposite of CBOW (Continuous Bag of Words):

- **CBOW**: "Given the words around me, what word am I?" (context → target)
- **Skip-gram**: "Given that I am this word, what words should be around me?" (target → context)

### Real-world Analogy:
Imagine you're reading a book and you see the word "coffee". A Skip-gram model would predict that words like "hot", "drink", "morning", "cup", or "caffeine" are likely to appear nearby in the same sentence or paragraph.

### How it Works:
1. **Input**: A target word (e.g., "coffee")
2. **Process**: Convert word to embedding vector, then predict probability distribution
3. **Output**: Probabilities for each word in vocabulary being a context word
4. **Training**: Learn by seeing actual word pairs from real text

Here's the enhanced documentation:

```python
class SkipGram_Model(nn.Module):
    """
    Skip-gram model for learning word embeddings through context prediction.
    
    The Skip-gram model learns word representations by training on the task of predicting 
    context words given a target word. For example, given the word "coffee", it learns 
    to predict words like "hot", "drink", "morning" that typically appear nearby.
    
    Architecture:
        1. Embedding Layer: Maps word indices to dense vector representations
        2. ReLU Activation: Adds non-linearity to the embeddings
        3. Linear Layer: Projects embeddings to vocabulary-sized output for prediction
    
    Training Process:
        - Input: Target word index (e.g., word_id for "coffee")
        - Output: Probability distribution over all vocabulary words
        - Loss: How well it predicts actual context words from training data
    
    Use Cases:
        - Learning semantic word relationships (king-queen, man-woman)
        - Finding synonyms and related words
        - Feature extraction for downstream NLP tasks
        - Word analogy tasks (king - man + woman = queen)
    """
    
    def __init__(self, vocab_size, embed_dim):
        """
        Initialize the Skip-gram model architecture.
        
        The model consists of:
        1. An embedding layer that converts word indices to dense vectors
        2. A linear layer that projects embeddings to vocabulary space for prediction
        
        Input:
            vocab_size (int): Total number of unique words in vocabulary
                             This determines both input size (word indices 0 to vocab_size-1)
                             and output size (predictions for each vocabulary word)
            embed_dim (int): Dimensionality of the learned word embeddings
                            Common values: 50, 100, 200, 300
                            Higher dims can capture more nuanced relationships
            
        Output:
            None (constructor)
            
        Example:
            >>> # Create model for vocabulary of 10,000 words with 100-dim embeddings
            >>> model = SkipGram_Model(vocab_size=10000, embed_dim=100)
            >>> print(f"Embedding matrix shape: {model.embeddings.weight.shape}")
            Embedding matrix shape: torch.Size([10000, 100])
            >>> print(f"Output layer shape: {model.fc.weight.shape}")  
            Output layer shape: torch.Size([10000, 100])
        """
        super(SkipGram_Model, self).__init__()
        
        # Embedding layer: Maps word indices to dense vectors
        # Each row represents the learned embedding for one word
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,  # Number of words (vocabulary size)
            embedding_dim=embed_dim     # Vector size for each word
        )
        
        # Linear projection layer: Maps embeddings back to vocabulary space
        # Used to predict context words by computing similarity scores
        self.fc = nn.Linear(
            in_features=embed_dim,      # Input: word embedding vector
            out_features=vocab_size     # Output: score for each vocabulary word
        )

    def forward(self, text):
        """
        Forward pass: predict context words given target words.
        
        Process Flow:
        1. Look up embeddings for input word indices
        2. Apply ReLU activation (adds non-linearity, helps with learning)
        3. Project to vocabulary space to get prediction scores
        4. Return logits (unnormalized probabilities) for each vocabulary word
        
        Input:
            text (torch.Tensor): Tensor of target word indices, shape (batch_size,)
                               Each element is an integer from 0 to vocab_size-1
                               representing a word in the vocabulary
            
        Output:
            torch.Tensor: Logits (prediction scores) for each vocabulary word
                         Shape: (batch_size, vocab_size)
                         Higher scores indicate higher probability of being context word
                         These logits can be converted to probabilities using softmax
            
        Example:
            >>> # Predict context for words "coffee" (id=42) and "morning" (id=156)
            >>> target_words = torch.tensor([42, 156])  # batch_size=2
            >>> logits = model(target_words)
            >>> print(logits.shape)
            torch.Size([2, 10000])  # 2 predictions, 10000 vocab words each
            >>> 
            >>> # Convert to probabilities and find most likely context words
            >>> probs = torch.softmax(logits, dim=1)
            >>> top_contexts = torch.topk(probs[0], k=5)  # Top 5 for "coffee"
            >>> print(f"Most likely context words for 'coffee': {top_contexts.indices}")
            Most likely context words for 'coffee': tensor([245, 1823, 67, 892, 334])
            # These indices would correspond to words like "hot", "drink", "morning", etc.
        """
        # Step 1: Convert word indices to embedding vectors
        # Shape: (batch_size,) -> (batch_size, embed_dim)
        out = self.embeddings(text)
        
        # Step 2: Apply ReLU activation function
        # Introduces non-linearity, helps model learn complex patterns
        # Negative values become 0, positive values stay unchanged
        out = torch.relu(out)
        
        # Step 3: Project embeddings to vocabulary space
        # Shape: (batch_size, embed_dim) -> (batch_size, vocab_size)
        # Each output represents "how likely is this vocabulary word to be context?"
        out = self.fc(out)
        
        return out
```

## Key Differences from CBOW:

| Aspect | CBOW | Skip-gram |
|--------|------|-----------|
| **Input** | Multiple context words | Single target word |
| **Output** | Single target word | Multiple context words |
| **Architecture** | Uses EmbeddingBag (averages contexts) | Uses regular Embedding |
| **Training Data** | (["the", "cat", "on"], "sat") | ("sat", "the"), ("sat", "cat"), ("sat", "on") |
| **Good For** | Frequent words, faster training | Rare words, better representations |

The Skip-gram model is particularly powerful because it generates multiple training examples from each sentence, making it excellent at learning representations for less frequent words!

