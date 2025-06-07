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
        
        # Try multiple file naming conventions - like checking different filing systems
        possible_files = [
            # Standard format
            (f'train.{self.src_lang}', f'train.{self.tgt_lang}'),
            # Alternative format  
            (f'train.{self.src_lang}-{self.tgt_lang}.{self.src_lang}', f'train.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}'),
            # Validation files might be named differently
            (f'val.{self.src_lang}', f'val.{self.tgt_lang}'),
            # Another common format
            (f'{self.src_lang}', f'{self.tgt_lang}'),
        ]
        
        src_file = None
        tgt_file = None
        
        # Find the first existing file pair
        for src_name, tgt_name in possible_files:
            src_candidate = os.path.join(data_path, src_name)
            tgt_candidate = os.path.join(data_path, tgt_name)
            
            if os.path.exists(src_candidate) and os.path.exists(tgt_candidate):
                src_file = src_candidate
                tgt_file = tgt_candidate
                print(f"Found data files: {src_name}, {tgt_name}")
                break
        
        if src_file is None or tgt_file is None:
            print(f"Data files not found in {data_path}")
            print("Looking for files with these patterns:")
            for src_name, tgt_name in possible_files:
                print(f"  - {src_name} and {tgt_name}")
            
            # List what files are actually there
            if os.path.exists(data_path):
                files = os.listdir(data_path)
                print(f"Files found in {data_path}: {files}")
            return []
        
        try:
            with open(src_file, 'r', encoding='utf-8') as sf, \
                 open(tgt_file, 'r', encoding='utf-8') as tf:
                # Read line by line - like reading parallel columns in a table
                for src_line, tgt_line in zip(sf, tf):
                    data.append((src_line.strip(), tgt_line.strip()))
                    
            print(f"Loaded {len(data)} parallel sentences from {data_path}")
        except FileNotFoundError as e:
            print(f"Error reading files: {e}")
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
    return torch.cat((torch.tensor([BOS_IDX], dtype=torch.long),
                      torch.tensor(token_ids, dtype=torch.long),
                      torch.tensor([EOS_IDX], dtype=torch.long)))

def tensor_transform_reversed(token_ids: List[int]) -> torch.Tensor:
    """Add BOS and EOS tokens with reversed sequence - like reading backwards"""
    return torch.cat((torch.tensor([BOS_IDX], dtype=torch.long),
                      torch.flip(torch.tensor(token_ids, dtype=torch.long), dims=(0,)),
                      torch.tensor([EOS_IDX], dtype=torch.long)))

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
        
        # Ensure integer dtype - like making sure all page numbers are whole numbers
        src_batch = src_batch.long()
        tgt_batch = tgt_batch.long()
        
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
    return " ".join([vocab.get_itos()[int(idx.item())] for idx in indices])

# Example usage and helper functions
def download_multi30k_dataset(data_dir: str = './data'):
    """Download and setup Multi30k dataset - like ordering and organizing your ingredients"""
    
    # URLs for the Multi30k dataset (same as in original torchtext code)
    urls = {
        'train': "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/training.tar.gz",
        'valid': "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/validation.tar.gz"
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    for split, url in urls.items():
        split_dir = os.path.join(data_dir, split)
        print(f"Setting up {split} data...")
        
        try:
            # Download and extract
            extract_path = download_and_extract_data(url, split_dir)
            
            # The extracted files might be in a subdirectory, let's find them
            extracted_files = []
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    if file.endswith('.de') or file.endswith('.en'):
                        extracted_files.append(os.path.join(root, file))
            
            if extracted_files:
                print(f"  Found {len(extracted_files)} files for {split}")
                
                # Move files to expected location if they're in subdirectories
                for file_path in extracted_files:
                    filename = os.path.basename(file_path)
                    target_path = os.path.join(split_dir, filename)
                    
                    if file_path != target_path:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        if not os.path.exists(target_path):
                            import shutil
                            shutil.copy2(file_path, target_path)
                            print(f"  Moved {filename} to {target_path}")
            else:
                print(f"  Warning: No .de or .en files found in {split_dir}")
                
        except Exception as e:
            print(f"Could not download {split} data: {e}")
            print("You may need to download Multi30k dataset manually")
            continue
    
    # Verify the setup
    expected_files = [
        os.path.join(data_dir, 'train', 'train.de'),
        os.path.join(data_dir, 'train', 'train.en'),
        os.path.join(data_dir, 'valid', 'val.de'),
        os.path.join(data_dir, 'valid', 'val.en')
    ]
    
    print("\nVerifying dataset setup...")
    for file_path in expected_files:
        if os.path.exists(file_path):
            line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            print(f"  ✓ {file_path} ({line_count} lines)")
        else:
            # Try alternative naming conventions
            alt_path = file_path.replace('train.', '').replace('val.', 'train.')
            if os.path.exists(alt_path):
                print(f"  ✓ Found alternative: {alt_path}")
            else:
                print(f"  ✗ Missing: {file_path}")
    
    print("Dataset setup complete!")

if __name__ == "__main__":
    # Example usage - like a recipe demonstration
    print("Setting up translation dataloaders without torchtext...")
    
    # First, download the dataset if it doesn't exist
    if not os.path.exists('./data/train') or not os.path.exists('./data/valid'):
        print("Dataset not found. Downloading Multi30k dataset...")
        download_multi30k_dataset('./data')
    else:
        print("Dataset directory exists. Checking for data files...")
        # Quick check to see if we have the expected files
        train_files = os.listdir('./data/train') if os.path.exists('./data/train') else []
        valid_files = os.listdir('./data/valid') if os.path.exists('./data/valid') else []
        
        if not any(f.endswith('.de') for f in train_files + valid_files):
            print("No data files found. Downloading...")
            download_multi30k_dataset('./data')
    
    # Create dataloaders
    train_loader, valid_loader, vocabs = get_translation_dataloaders_no_torchtext(
        batch_size=4, 
        flip=False
    )
    
    if train_loader is not None:
        # Test the dataloader - like tasting your cooking
        print("\nTesting dataloader...")
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
    else:
        print("Failed to create dataloaders. Please check the data setup.")

