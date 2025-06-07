import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
import tarfile
import urllib.request
from collections import Counter
import re
from typing import List, Tuple, Dict

# Special token indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

class SimpleVocab:
    """A simple vocabulary class - like a dictionary that maps words to numbers"""
    def __init__(self, tokens: List[str], min_freq: int = 1):
        # Count word frequencies - like counting how often each ingredient appears in recipes
        counter = Counter(tokens)
        
        # Filter by minimum frequency - removing rare ingredients
        filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
        
        # Build vocabulary: special tokens first, then regular tokens
        self.vocab = special_symbols + sorted(set(filtered_tokens))
        
        # Create mappings - like having both a phone book and reverse phone book
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Set default index for unknown tokens
        self.default_index = UNK_IDX
    
    def __call__(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices - like translating words to numbers"""
        return [self.token_to_idx.get(token, self.default_index) for token in tokens]
    
    def get_itos(self) -> List[str]:
        """Get index-to-string mapping - like a numbered list of words"""
        return self.vocab
    
    def __len__(self):
        return len(self.vocab)

class SimpleTokenizer:
    """A basic tokenizer - like a text splitter that understands punctuation"""
    def __init__(self, language: str = 'en'):
        self.language = language
    
    def __call__(self, text: str) -> List[str]:
        """Tokenize text - split sentences into words like cutting a sentence into pieces"""
        # Simple tokenization: lowercase, split on whitespace and punctuation
        text = text.lower().strip()
        # Keep punctuation separate - like separating words from commas and periods
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        tokens = text.split()
        return [token for token in tokens if token.strip()]

class Multi30kDataset(Dataset):
    """Custom dataset class - like a recipe book that knows how to serve up data"""
    def __init__(self, data_path: str, src_lang: str = 'de', tgt_lang: str = 'en'):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Tuple[str, str]]:
        """Load parallel text data - like reading a bilingual dictionary"""
        data = []
        
        # File paths - like knowing where your German and English books are
        src_file = os.path.join(data_path, f'train.{self.src_lang}')
        tgt_file = os.path.join(data_path, f'train.{self.tgt_lang}')
        
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            # Try alternative file structure
            src_file = os.path.join(data_path, f'train.{self.src_lang}-{self.tgt_lang}.{self.src_lang}')
            tgt_file = os.path.join(data_path, f'train.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}')
        
        try:
            with open(src_file, 'r', encoding='utf-8') as sf, \
                 open(tgt_file, 'r', encoding='utf-8') as tf:
                # Read line by line - like reading parallel columns in a table
                for src_line, tgt_line in zip(sf, tf):
                    data.append((src_line.strip(), tgt_line.strip()))
        except FileNotFoundError:
            print(f"Data files not found. Please download and extract the Multi30k dataset to {data_path}")
            return []
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def download_and_extract_data(url: str, extract_path: str) -> str:
    """Download and extract dataset - like ordering and unpacking a delivery"""
    os.makedirs(extract_path, exist_ok=True)
    
    # Download file
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_path, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    
    # Extract if it's a tar file
    if filename.endswith('.tar.gz'):
        print(f"Extracting {filename}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(extract_path)
    
    return extract_path

def build_vocab(dataset: List[Tuple[str, str]], tokenizers: Dict[str, SimpleTokenizer], 
                src_lang: str, tgt_lang: str, min_freq: int = 1) -> Dict[str, SimpleVocab]:
    """Build vocabularies for both languages - like creating dictionaries for translation"""
    
    # Collect all tokens - like gathering all unique words from both languages
    src_tokens = []
    tgt_tokens = []
    
    for src_text, tgt_text in dataset:
        src_tokens.extend(tokenizers[src_lang](src_text))
        tgt_tokens.extend(tokenizers[tgt_lang](tgt_text))
    
    # Build vocabularies - like creating word-to-number mappings
    vocabs = {
        src_lang: SimpleVocab(src_tokens, min_freq),
        tgt_lang: SimpleVocab(tgt_tokens, min_freq)
    }
    
    return vocabs

def tensor_transform_normal(token_ids: List[int]) -> torch.Tensor:
    """Add BOS and EOS tokens - like adding quotation marks around a sentence"""
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def tensor_transform_reversed(token_ids: List[int]) -> torch.Tensor:
    """Add BOS and EOS tokens with reversed sequence - like reading backwards"""
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.flip(torch.tensor(token_ids), dims=(0,)),
                      torch.tensor([EOS_IDX])))

class TranslationCollator:
    """Collate function - like a chef who prepares batches of ingredients together"""
    def __init__(self, tokenizers: Dict[str, SimpleTokenizer], 
                 vocabs: Dict[str, SimpleVocab], 
                 src_lang: str, tgt_lang: str, 
                 device: torch.device, flip_src: bool = False):
        self.tokenizers = tokenizers
        self.vocabs = vocabs
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device
        self.flip_src = flip_src
    
    def __call__(self, batch: List[Tuple[str, str]]):
        """Process a batch of data - like preparing multiple dishes at once"""
        src_batch, tgt_batch = [], []
        
        for src_text, tgt_text in batch:
            # Tokenize - split into words
            src_tokens = self.tokenizers[self.src_lang](src_text)
            tgt_tokens = self.tokenizers[self.tgt_lang](tgt_text)
            
            # Convert to indices - translate words to numbers
            src_indices = self.vocabs[self.src_lang](src_tokens)
            tgt_indices = self.vocabs[self.tgt_lang](tgt_tokens)
            
            # Add special tokens and convert to tensors
            if self.flip_src:
                src_tensor = tensor_transform_reversed(src_indices)
            else:
                src_tensor = tensor_transform_normal(src_indices)
            
            tgt_tensor = tensor_transform_normal(tgt_indices)
            
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
        
        # Pad sequences to same length - like making all books the same thickness
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        
        # Transpose for sequence-first format - like rotating a table
        src_batch = src_batch.t()
        tgt_batch = tgt_batch.t()
        
        return src_batch.to(self.device), tgt_batch.to(self.device)

def get_translation_dataloaders_no_torchtext(data_path: str = './data', 
                                           batch_size: int = 4, 
                                           flip: bool = False,
                                           src_lang: str = 'de',
                                           tgt_lang: str = 'en'):
    """
    Create dataloaders without torchtext - like setting up a translation assembly line
    
    Example:
        train_loader, valid_loader, vocabs = get_translation_dataloaders_no_torchtext()
        for src_batch, tgt_batch in train_loader:
            # Your training code here
            pass
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizers - like having translators for each language
    tokenizers = {
        src_lang: SimpleTokenizer(src_lang),
        tgt_lang: SimpleTokenizer(tgt_lang)
    }
    
    # Create datasets - like organizing your data library
    train_dataset = Multi30kDataset(os.path.join(data_path, 'train'), src_lang, tgt_lang)
    valid_dataset = Multi30kDataset(os.path.join(data_path, 'valid'), src_lang, tgt_lang)
    
    if len(train_dataset) == 0:
        print("No training data found. You may need to download the Multi30k dataset manually.")
        print("Expected file structure:")
        print(f"  {data_path}/train/train.{src_lang}")
        print(f"  {data_path}/train/train.{tgt_lang}")
        print(f"  {data_path}/valid/train.{src_lang}")
        print(f"  {data_path}/valid/train.{tgt_lang}")
        return None, None, None
    
    # Build vocabularies - create word dictionaries from training data
    print("Building vocabularies...")
    vocabs = build_vocab(train_dataset.data, tokenizers, src_lang, tgt_lang)
    
    print(f"Source vocabulary size: {len(vocabs[src_lang])}")
    print(f"Target vocabulary size: {len(vocabs[tgt_lang])}")
    
    # Sort datasets by source length - like organizing books by thickness
    train_data_sorted = sorted(train_dataset.data, key=lambda x: len(tokenizers[src_lang](x[0])))
    valid_data_sorted = sorted(valid_dataset.data, key=lambda x: len(tokenizers[src_lang](x[0])))
    
    # Create collate function - the batch processor
    collate_fn = TranslationCollator(tokenizers, vocabs, src_lang, tgt_lang, device, flip)
    
    # Create dataloaders - like conveyor belts for your data
    train_dataloader = DataLoader(train_data_sorted, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                drop_last=True,
                                shuffle=False)  # Keep sorted order
    
    valid_dataloader = DataLoader(valid_data_sorted, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                drop_last=True,
                                shuffle=False)
    
    return train_dataloader, valid_dataloader, vocabs

def index_to_text(indices: torch.Tensor, vocab: SimpleVocab) -> str:
    """Convert indices back to text - like translating numbers back to words"""
    return " ".join([vocab.get_itos()[idx.item()] for idx in indices])

# Example usage and helper functions
def download_multi30k_sample():
    """Download a sample of Multi30k data for testing"""
    # Note: You'll need to implement actual download URLs or provide your own data
    # This is a placeholder showing the expected structure
    urls = {
        'train': "https://your-data-source.com/training.tar.gz",
        'valid': "https://your-data-source.com/validation.tar.gz"
    }
    
    for split, url in urls.items():
        try:
            download_and_extract_data(url, f'./data/{split}')
        except Exception as e:
            print(f"Could not download {split} data: {e}")
            print("Please download Multi30k dataset manually and place in ./data/ directory")

if __name__ == "__main__":
    # Example usage - like a recipe demonstration
    print("Setting up translation dataloaders without torchtext...")
    
    # Create dataloaders
    train_loader, valid_loader, vocabs = get_translation_dataloaders_no_torchtext(
        batch_size=4, 
        flip=False
    )
    
    if train_loader is not None:
        # Test the dataloader - like tasting your cooking
        print("Testing dataloader...")
        for i, (src_batch, tgt_batch) in enumerate(train_loader):
            print(f"Batch {i+1}:")
            print(f"  Source shape: {src_batch.shape}")
            print(f"  Target shape: {tgt_batch.shape}")
            
            # Show first example in batch
            src_text = index_to_text(src_batch[:, 0], vocabs['de'])
            tgt_text = index_to_text(tgt_batch[:, 0], vocabs['en'])
            print(f"  Sample source: {src_text}")
            print(f"  Sample target: {tgt_text}")
            
            if i >= 2:  # Just show first few batches
                break
