**Utility Functions:**
- `basic_english_tokenizer`: Cleans and splits text (like turning "Hello, world!" into ['hello', 'world'])
- `plot_embeddings_headless`: Creates t-SNE visualizations without display
- `find_similar_words`: Finds semantically similar words using cosine similarity
- `train_model`: Universal training loop that works with both CBOW and Skip-gram

**Data Processing Functions:**
- `tokenize_sentences`: Processes multi-line text into tokenized sentences
- `collate_batch`: Prepares CBOW data for EmbeddingBag (handles variable context sizes)
- `collate_fn`: Prepares Skip-gram data for standard Embedding layer

## Think of it Like a Recipe Book:

Each function is now like a detailed recipe that tells you:
- **What dish you're making** (purpose)
- **What ingredients you need** (inputs) 
- **What you'll get at the end** (outputs)
- **A step-by-step example** (usage example)

For instance, the `find_similar_words` function is like a recipe for "finding words that taste similar to your target word" - you give it a word like "king", and it returns words like "queen", "prince", "royal" based on their embedding similarity.

The documentation now follows Python best practices with detailed docstrings that any developer (or future you!) can easily understand and use. Each example shows realistic input/output shapes and values, making it clear how to use each component in the word embedding pipeline.
