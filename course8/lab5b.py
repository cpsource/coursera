# Modernized Word2Vec Implementation - Headless Version

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
from gensim.models import Word2Vec
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import re
import io
import base64

# Suppress warnings
def warn(*args, **kwargs):
    """
    Override function to suppress warning messages.
    
    Input: 
        *args: Variable length argument list (ignored)
        **kwargs: Arbitrary keyword arguments (ignored)
    
    Output: 
        None (does nothing)
    
    Example:
        # Used to replace the default warning function
        warnings.warn = warn
    """
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

def basic_english_tokenizer(text):
    """
    Custom tokenizer that processes English text by splitting on whitespace and removing punctuation.
    
    Input:
        text (str): Raw text string to tokenize
        
    Output:
        list: List of cleaned tokens (lowercase, alphabetic only)
        
    Example:
        >>> text = "Hello, world! How are you?"
        >>> tokens = basic_english_tokenizer(text)
        >>> print(tokens)
        ['hello', 'world', 'how', 'are', 'you']
    """
    # Convert to lowercase and split on whitespace
    tokens = text.lower().split()
    # Remove punctuation and keep only alphabetic characters
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if token.strip()]
    # Filter out empty tokens
    return [token for token in tokens if token]

class Vocabulary:
    """
    Custom vocabulary class that manages token-to-index and index-to-token mappings.
    Replaces the deprecated torchtext vocabulary functionality.
    """
    
    def __init__(self, specials=None):
        """
        Initialize the vocabulary with optional special tokens.
        
        Input:
            specials (list, optional): List of special tokens like ['<unk>', '<pad>']
            
        Output:
            None (constructor)
            
        Example:
            >>> vocab = Vocabulary(specials=['<unk>', '<pad>'])
            >>> print(vocab.specials)
            ['<unk>', '<pad>']
        """
        self.specials = specials or ['<unk>']
        self.token_to_idx = {}
        self.idx_to_token = []
        self.default_index = 0
        
    def build_from_iterator(self, token_iterator):
        """
        Build vocabulary from an iterator of tokenized sentences.
        
        Input:
            token_iterator: Iterator that yields lists of tokens
            
        Output:
            None (modifies internal state)
            
        Example:
            >>> sentences = [['hello', 'world'], ['how', 'are', 'you']]
            >>> vocab.build_from_iterator(sentences)
            >>> print(len(vocab))
            6  # <unk> + 5 unique words
        """
        # Add special tokens first
        for special in self.specials:
            if special not in self.token_to_idx:
                self.token_to_idx[special] = len(self.idx_to_token)
                self.idx_to_token.append(special)
        
        # Add tokens from iterator
        for tokens in token_iterator:
            for token in tokens:
                if token not in self.token_to_idx:
                    self.token_to_idx[token] = len(self.idx_to_token)
                    self.idx_to_token.append(token)
    
    def __getitem__(self, token):
        """
        Get the index for a given token.
        
        Input:
            token (str): Token to look up
            
        Output:
            int: Index of the token, or default_index if not found
            
        Example:
            >>> idx = vocab['hello']
            >>> print(idx)
            1  # assuming 'hello' is the second token after <unk>
        """
        return self.token_to_idx.get(token, self.default_index)
    
    def __len__(self):
        """
        Get the size of the vocabulary.
        
        Input:
            None
            
        Output:
            int: Number of tokens in vocabulary
            
        Example:
            >>> size = len(vocab)
            >>> print(size)
            100
        """
        return len(self.idx_to_token)
    
    def get_itos(self):
        """
        Get index-to-string mapping (list of tokens ordered by index).
        
        Input:
            None
            
        Output:
            list: List of tokens ordered by their indices
            
        Example:
            >>> tokens = vocab.get_itos()
            >>> print(tokens[:3])
            ['<unk>', 'hello', 'world']
        """
        return self.idx_to_token
    
    def get_stoi(self):
        """
        Get string-to-index mapping (dictionary).
        
        Input:
            None
            
        Output:
            dict: Dictionary mapping tokens to their indices
            
        Example:
            >>> mapping = vocab.get_stoi()
            >>> print(mapping['hello'])
            1
        """
        return self.token_to_idx
    
    def set_default_index(self, index):
        """
        Set the default index for unknown tokens.
        
        Input:
            index (int): Index to use for unknown tokens
            
        Output:
            None
            
        Example:
            >>> vocab.set_default_index(0)  # Use index 0 for <unk>
        """
        self.default_index = index

def plot_embeddings_headless(word_embeddings, vocab, save_path='word_embeddings.png'):
    """
    Create a t-SNE visualization of word embeddings and save to file (headless mode).
    
    Input:
        word_embeddings (numpy.ndarray): 2D array of word embeddings, shape (vocab_size, embed_dim)
        vocab (Vocabulary): Vocabulary object containing word-to-index mappings
        save_path (str): File path to save the plot image
        
    Output:
        None (saves plot to file)
        
    Example:
        >>> embeddings = np.random.randn(100, 50)  # 100 words, 50-dim embeddings
        >>> plot_embeddings_headless(embeddings, vocab, 'my_embeddings.png')
        Plot saved to my_embeddings.png
    """
    tsne = TSNE(n_components=2, random_state=0)
    word_embeddings_2d = tsne.fit_transform(word_embeddings)

    # Create the plot
    plt.figure(figsize=(15, 15))
    for i, word in enumerate(vocab.get_itos()):
        plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
        plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))

    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title("Word Embeddings visualized with t-SNE")
    
    # Save plot instead of showing
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved to {save_path}")

def save_plot_headless(epoch_losses, save_path='training_loss.png', title='Training Loss'):
    """
    Create and save a line plot of training losses over epochs (headless mode).
    
    Input:
        epoch_losses (list): List of loss values for each epoch
        save_path (str): File path to save the plot image
        title (str): Title for the plot
        
    Output:
        None (saves plot to file)
        
    Example:
        >>> losses = [0.8, 0.6, 0.4, 0.3, 0.2]
        >>> save_plot_headless(losses, 'training.png', 'My Model Loss')
        Plot saved to training.png
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def find_similar_words(word, word_embeddings, vocab, top_k=5):
    """
    Find the most similar words to a target word using cosine similarity of embeddings.
    
    Input:
        word (str): Target word to find similarities for
        word_embeddings (numpy.ndarray): 2D array of word embeddings
        vocab (Vocabulary): Vocabulary object for word-index mapping
        top_k (int): Number of most similar words to return
        
    Output:
        list: List of the top_k most similar words (strings)
        
    Example:
        >>> similar = find_similar_words('king', embeddings, vocab, top_k=3)
        >>> print(similar)
        ['queen', 'prince', 'royal']
    """
    if word not in vocab.get_stoi():
        print("Word not found in vocabulary.")
        return []

    # Get the embedding for the given word
    word_index = vocab.get_stoi()[word]
    target_embedding = torch.tensor(word_embeddings[word_index])

    # Calculate cosine similarities
    similarities = {}
    for i, w in enumerate(vocab.get_itos()):
        if w != word:
            embedding = torch.tensor(word_embeddings[i])
            similarity = torch.dot(target_embedding, embedding) / (
                torch.norm(target_embedding) * torch.norm(embedding)
            )
            similarities[w] = similarity.item()

    # Sort and return top k
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar_words = [w for w, _ in sorted_similarities[:top_k]]
    return most_similar_words

def train_model(model, dataloader, criterion, optimizer, num_epochs=1000):
    """
    Train a PyTorch model for word embeddings (supports both CBOW and Skip-gram architectures).
    
    Input:
        model (nn.Module): PyTorch model to train (CBOW or Skip-gram)
        dataloader (DataLoader): DataLoader providing training batches
        criterion (nn.Module): Loss function (typically CrossEntropyLoss)
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights
        num_epochs (int): Number of training epochs
        
    Output:
        tuple: (trained_model, epoch_losses)
            - trained_model (nn.Module): The trained model
            - epoch_losses (list): List of average loss per epoch
            
    Example:
        >>> model = CBOW(vocab_size=1000, embed_dim=100, num_class=1000)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> trained_model, losses = train_model(model, dataloader, criterion, optimizer, 100)
        >>> print(f"Final loss: {losses[-1]:.4f}")
        Final loss: 0.2543
    """
    epoch_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        running_loss = 0.0

        for idx, samples in enumerate(dataloader):
            optimizer.zero_grad()

            # Check for EmbeddingBag layer in the model
            if any(isinstance(module, nn.EmbeddingBag) for _, module in model.named_modules()):
                target, context, offsets = samples
                predicted = model(context, offsets)
            # Check for Embedding layer in the model
            elif any(isinstance(module, nn.Embedding) for _, module in model.named_modules()):
                target, context = samples
                predicted = model(context)

            loss = criterion(predicted, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running_loss += loss.item()

        # Append average loss for the epoch
        epoch_losses.append(running_loss / len(dataloader))

    return model, epoch_losses

# Sample data
toy_data = """I wish I was little bit taller
I wish I was a baller
She wore a small black dress to the party
The dog chased a big red ball in the park
He had a huge smile on his face when he won the race
The tiny kitten played with a fluffy toy mouse
The team celebrated their victory with a grand parade
She bought a small, delicate necklace for her sister
The mountain peak stood majestic and tall against the clear blue sky
The toddler took small, careful steps as she learned to walk
The house had a spacious backyard with a big swimming pool
He felt a sense of accomplishment after completing the challenging puzzle
The chef prepared a delicious, flavorful dish using fresh ingredients
The children played happily in the small, cozy room
The book had an enormous impact on readers around the world
The wind blew gently, rustling the leaves of the tall trees
She painted a beautiful, intricate design on the small canvas
The concert hall was filled with thousands of excited fans
The garden was adorned with colorful flowers of all sizes
I hope to achieve great success in my chosen career path
The skyscraper towered above the city, casting a long shadow
He gazed in awe at the breathtaking view from the mountaintop
The artist created a stunning masterpiece with bold brushstrokes
The baby took her first steps, a small milestone that brought joy to her parents
The team put in a tremendous amount of effort to win the championship
The sun set behind the horizon, painting the sky in vibrant colors
The professor gave a fascinating lecture on the history of ancient civilizations
The house was filled with laughter and the sound of children playing
She received a warm, enthusiastic welcome from the audience
The marathon runner had incredible endurance and determination
The child's eyes sparkled with excitement upon opening the gift
The ship sailed across the vast ocean, guided by the stars
The company achieved remarkable growth in a short period of time
The team worked together harmoniously to complete the project
The puppy wagged its tail, expressing its happiness and affection
She wore a stunning gown that made her feel like a princess
The building had a grand entrance with towering columns
The concert was a roaring success, with the crowd cheering and clapping
The baby took a tiny bite of the sweet, juicy fruit
The athlete broke a new record, achieving a significant milestone in her career
The sculpture was a masterpiece of intricate details and craftsmanship
The forest was filled with towering trees, creating a sense of serenity
The children built a small sandcastle on the beach, their imaginations running wild
The mountain range stretched as far as the eye could see, majestic and awe-inspiring
The artist's brush glided smoothly across the canvas, creating a beautiful painting
She received a small token of appreciation for her hard work and dedication
The orchestra played a magnificent symphony that moved the audience to tears
The flower bloomed in vibrant colors, attracting butterflies and bees
The team celebrated their victory with a big, extravagant party
The child's laughter echoed through the small room, filling it with joy
The sunflower stood tall, reaching for the sky with its bright yellow petals
The city skyline was dominated by tall buildings and skyscrapers
The cake was adorned with a beautiful, elaborate design for the special occasion
The storm brought heavy rain and strong winds, causing widespread damage
The small boat sailed peacefully on the calm, glassy lake
The artist used bold strokes of color to create a striking and vivid painting
The couple shared a passionate kiss under the starry night sky
The mountain climber reached the summit after a long and arduous journey
The child's eyes widened in amazement as the magician performed his tricks
The garden was filled with the sweet fragrance of blooming flowers
The basketball player made a big jump and scored a spectacular slam dunk
The cat pounced on a small mouse, displaying its hunting instincts
The mansion had a grand entrance with a sweeping staircase and chandeliers
The raindrops fell gently, creating a rhythmic patter on the roof
The baby took a big step forward, encouraged by her parents' applause
The actor delivered a powerful and emotional performance on stage
The butterfly fluttered its delicate wings, mesmerizing those who watched
The company launched a small-scale advertising campaign to test the market
The building was constructed with strong, sturdy materials to withstand earthquakes
The singer's voice was powerful and resonated throughout the concert hall
The child built a massive sandcastle with towers, moats, and bridges
The garden was teeming with a variety of small insects and buzzing bees
The athlete's muscles were well-developed and strong from years of training
The sun cast long shadows as it set behind the mountains
The couple exchanged heartfelt vows in a beautiful, intimate ceremony
The dog wagged its tail vigorously, a sign of excitement and happiness
The baby let out a tiny giggle, bringing joy to everyone around"""

# Tokenize the data
tokenized_toy_data = basic_english_tokenizer(toy_data)

# Build vocabulary
def tokenize_sentences(text):
    """
    Generator function that yields tokenized sentences from multi-line text.
    
    Input:
        text (str): Multi-line text string with sentences separated by newlines
        
    Output:
        generator: Yields lists of tokens for each sentence
        
    Example:
        >>> text = "Hello world.\nHow are you?\nI am fine."
        >>> for tokens in tokenize_sentences(text):
        ...     print(tokens)
        ['hello', 'world']
        ['how', 'are', 'you']
        ['i', 'am', 'fine']
    """
    sentences = text.split('\n')
    for sentence in sentences:
        yield basic_english_tokenizer(sentence)

vocab = Vocabulary(specials=['<unk>'])
vocab.build_from_iterator(tokenize_sentences(toy_data))
vocab.set_default_index(vocab['<unk>'])

# Test tokenization
sample_sentence = "I wish I was a baller"
tokenized_sample = basic_english_tokenizer(sample_sentence)
encoded_sample = [vocab[token] for token in tokenized_sample]
print("Encoded sample:", encoded_sample)

text_pipeline = lambda tokens: [vocab[token] for token in tokens]

# CBOW Data Preparation
CONTEXT_SIZE = 2
cbow_data = []

for i in range(CONTEXT_SIZE, len(tokenized_toy_data) - CONTEXT_SIZE):
    context = (
        [tokenized_toy_data[i - CONTEXT_SIZE + j] for j in range(CONTEXT_SIZE)]
        + [tokenized_toy_data[i + j + 1] for j in range(CONTEXT_SIZE)]
    )
    target = tokenized_toy_data[i]
    cbow_data.append((context, target))

print("CBOW data sample:", cbow_data[0])
print("CBOW data sample:", cbow_data[1])

def collate_batch(batch):
    """
    Collate function for CBOW model data batching.
    Converts a batch of (context, target) pairs into tensors suitable for EmbeddingBag layer.
    
    Input:
        batch (list): List of tuples containing (context_words, target_word) pairs
                     where context_words is a list of words surrounding the target
                     
    Output:
        tuple: (target_list, context_list, offsets)
            - target_list (torch.Tensor): Tensor of target word indices, shape (batch_size,)
            - context_list (torch.Tensor): Flattened tensor of all context word indices
            - offsets (torch.Tensor): Starting positions of each sample's context in context_list
            
    Example:
        >>> batch = [(['the', 'cat'], 'sat'), (['cat', 'on'], 'sat')]
        >>> targets, contexts, offs = collate_batch(batch)
        >>> print(targets.shape, contexts.shape, offs.shape)
        torch.Size([2]) torch.Size([4]) torch.Size([2])
    """
    target_list, context_list, offsets = [], [], [0]
    for _context, _target in batch:
        target_list.append(vocab[_target])
        processed_context = torch.tensor(text_pipeline(_context), dtype=torch.int64)
        context_list.append(processed_context)
        offsets.append(processed_context.size(0))
    
    target_list = torch.tensor(target_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    context_list = torch.cat(context_list)
    return target_list.to(device), context_list.to(device), offsets.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Test collate function
target_list, context_list, offsets = collate_batch(cbow_data[0:10])
print(f"Target list: {target_list}")
print(f"Context list: {context_list}")
print(f"Offsets: {offsets}")

BATCH_SIZE = 64
dataloader_cbow = DataLoader(
    cbow_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

class CBOW(nn.Module):
    """
    Continuous Bag of Words (CBOW) model for word embeddings.
    Predicts a target word from its surrounding context words.
    """
    
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        Initialize the CBOW model architecture.
        
        Input:
            vocab_size (int): Size of the vocabulary (number of unique words)
            embed_dim (int): Dimensionality of word embeddings
            num_class (int): Number of output classes (typically same as vocab_size)
            
        Output:
            None (constructor)
            
        Example:
            >>> model = CBOW(vocab_size=1000, embed_dim=100, num_class=1000)
            >>> print(model.embedding.num_embeddings)
            1000
        """
        super(CBOW, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.linear1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc = nn.Linear(embed_dim//2, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize model weights with uniform distribution.
        
        Input:
            None
            
        Output:
            None (modifies model weights in-place)
            
        Example:
            >>> model = CBOW(1000, 100, 1000)
            >>> model.init_weights()  # Called automatically in __init__
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, text, offsets):
        """
        Forward pass through the CBOW model.
        
        Input:
            text (torch.Tensor): Flattened tensor of context word indices
            offsets (torch.Tensor): Starting positions for each sample's context
            
        Output:
            torch.Tensor: Logits for each word in vocabulary, shape (batch_size, vocab_size)
            
        Example:
            >>> text = torch.tensor([1, 2, 3, 4])  # Context words
            >>> offsets = torch.tensor([0, 2])     # Two samples: [1,2] and [3,4]
            >>> output = model(text, offsets)
            >>> print(output.shape)
            torch.Size([2, 1000])  # batch_size=2, vocab_size=1000
        """
        out = self.embedding(text, offsets)
        out = torch.relu(self.linear1(out))
        return self.fc(out)

# Initialize CBOW model
vocab_size = len(vocab)
emsize = 24
model_cbow = CBOW(vocab_size, emsize, vocab_size).to(device)

# Training setup
LR = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_cbow.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# Train CBOW model
print("Training CBOW model...")
model_cbow, epoch_losses = train_model(model_cbow, dataloader_cbow, criterion, optimizer, num_epochs=400)

# Save training loss plot
save_plot_headless(epoch_losses, 'cbow_training_loss.png', 'CBOW Training Loss')

# Extract and visualize CBOW embeddings
word_embeddings = model_cbow.embedding.weight.detach().cpu().numpy()

word = 'baller'
if word in vocab.get_stoi():
    word_index = vocab.get_stoi()[word]
    print(f"Embedding for '{word}':", word_embeddings[word_index])

plot_embeddings_headless(word_embeddings, vocab, 'cbow_embeddings.png')

# Skip-gram Data Preparation
skip_data = []

for i in range(CONTEXT_SIZE, len(tokenized_toy_data) - CONTEXT_SIZE):
    context = (
        [tokenized_toy_data[i - j - 1] for j in range(CONTEXT_SIZE)]
        + [tokenized_toy_data[i + j + 1] for j in range(CONTEXT_SIZE)]
    )
    target = tokenized_toy_data[i]
    skip_data.append((target, context))

skip_data_ = [[(sample[0], word) for word in sample[1]] for sample in skip_data]
skip_data_flat = [item for items in skip_data_ for item in items]

print("Skip-gram data sample:", skip_data_flat[8:28])

def collate_fn(batch):
    """Collate function for Skip-gram data"""
    target_list, context_list = [], []
    for _context, _target in batch:
        target_list.append(vocab[_target])
        context_list.append(vocab[_context])
    
    target_list = torch.tensor(target_list, dtype=torch.int64)
    context_list = torch.tensor(context_list, dtype=torch.int64)
    return target_list.to(device), context_list.to(device)

dataloader = DataLoader(skip_data_flat, batch_size=BATCH_SIZE, collate_fn=collate_fn)

class SkipGram_Model(nn.Module):
    """Skip-gram model implementation"""
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.fc = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, text):
        out = self.embeddings(text)
        out = torch.relu(out)
        out = self.fc(out)
        return out

# Initialize Skip-gram model
model_sg = SkipGram_Model(vocab_size, emsize).to(device)

# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_sg.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# Train Skip-gram model
print("Training Skip-gram model...")
model_sg, epoch_losses = train_model(model_sg, dataloader, criterion, optimizer, num_epochs=400)

# Save training loss plot
save_plot_headless(epoch_losses, 'skipgram_training_loss.png', 'Skip-gram Training Loss')

# Extract and visualize Skip-gram embeddings
word_embeddings = model_sg.embeddings.weight.detach().cpu().numpy()
plot_embeddings_headless(word_embeddings, vocab, 'skipgram_embeddings.png')

# Test similarity function
print("\nSimilar words to 'small':")
similar_words = find_similar_words('small', word_embeddings, vocab, top_k=5)
print(similar_words)

print("\nSimilar words to 'big':")
similar_words = find_similar_words('big', word_embeddings, vocab, top_k=5)
print(similar_words)

print("\nTraining complete! Check the generated PNG files for visualizations.")
