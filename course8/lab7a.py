# Sequence-to-Sequence RNN Models: Translation Task without torchtext

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import os
import tarfile
import urllib.request
from collections import Counter
import re
from typing import List, Tuple, Dict
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import random
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Special token indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

class SimpleVocab:
    """A simple vocabulary class - like a dictionary that maps words to numbers"""
    def __init__(self, tokens: List[str], min_freq: int = 1):
        # Count word frequencies
        counter = Counter(tokens)
        
        # Filter by minimum frequency
        filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
        
        # Build vocabulary: special tokens first, then regular tokens
        self.vocab = special_symbols + sorted(set(filtered_tokens))
        
        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Set default index for unknown tokens
        self.default_index = UNK_IDX
    
    def __call__(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices"""
        return [self.token_to_idx.get(token, self.default_index) for token in tokens]
    
    def get_itos(self) -> List[str]:
        """Get index-to-string mapping"""
        return self.vocab
    
    def get_stoi(self) -> Dict[str, int]:
        """Get string-to-index mapping - torchtext compatibility"""
        return self.token_to_idx
    
    def __len__(self):
        return len(self.vocab)

class SimpleTokenizer:
    """A basic tokenizer"""
    def __init__(self, language: str = 'en'):
        self.language = language
    
    def __call__(self, text: str) -> List[str]:
        """Tokenize text"""
        text = text.lower().strip()
        text = re.sub(r'([.!?,:;])', r' \\1 ', text)
        tokens = text.split()
        return [token for token in tokens if token.strip()]

class Multi30kDataset(Dataset):
    """Custom dataset class"""
    def __init__(self, data_path: str, src_lang: str = 'de', tgt_lang: str = 'en'):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Tuple[str, str]]:
        """Load parallel text data"""
        data = []
        
        # Try multiple file naming conventions
        possible_files = [
            (f'train.{self.src_lang}', f'train.{self.tgt_lang}'),
            (f'train.{self.src_lang}-{self.tgt_lang}.{self.src_lang}', f'train.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}'),
            (f'val.{self.src_lang}', f'val.{self.tgt_lang}'),
            (f'{self.src_lang}', f'{self.tgt_lang}'),
        ]
        
        src_file = None
        tgt_file = None
        
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
            return []
        
        try:
            with open(src_file, 'r', encoding='utf-8') as sf, \
                 open(tgt_file, 'r', encoding='utf-8') as tf:
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
    """Download and extract dataset"""
    os.makedirs(extract_path, exist_ok=True)
    
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_path, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    
    if filename.endswith('.tar.gz'):
        print(f"Extracting {filename}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(extract_path)
    
    return extract_path

def build_vocab(dataset: List[Tuple[str, str]], tokenizers: Dict[str, SimpleTokenizer], 
                src_lang: str, tgt_lang: str, min_freq: int = 1) -> Dict[str, SimpleVocab]:
    """Build vocabularies for both languages"""
    
    src_tokens = []
    tgt_tokens = []
    
    for src_text, tgt_text in dataset:
        src_tokens.extend(tokenizers[src_lang](src_text))
        tgt_tokens.extend(tokenizers[tgt_lang](tgt_text))
    
    vocabs = {
        src_lang: SimpleVocab(src_tokens, min_freq),
        tgt_lang: SimpleVocab(tgt_tokens, min_freq)
    }
    
    return vocabs

def tensor_transform_normal(token_ids: List[int]) -> torch.Tensor:
    """Add BOS and EOS tokens"""
    return torch.cat((torch.tensor([BOS_IDX], dtype=torch.long),
                      torch.tensor(token_ids, dtype=torch.long),
                      torch.tensor([EOS_IDX], dtype=torch.long)))

def tensor_transform_reversed(token_ids: List[int]) -> torch.Tensor:
    """Add BOS and EOS tokens with reversed sequence"""
    return torch.cat((torch.tensor([BOS_IDX], dtype=torch.long),
                      torch.flip(torch.tensor(token_ids, dtype=torch.long), dims=(0,)),
                      torch.tensor([EOS_IDX], dtype=torch.long)))

class TranslationCollator:
    """Collate function for batching"""
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
        """Process a batch of data"""
        src_batch, tgt_batch = [], []
        
        for src_text, tgt_text in batch:
            src_tokens = self.tokenizers[self.src_lang](src_text)
            tgt_tokens = self.tokenizers[self.tgt_lang](tgt_text)
            
            src_indices = self.vocabs[self.src_lang](src_tokens)
            tgt_indices = self.vocabs[self.tgt_lang](tgt_tokens)
            
            if self.flip_src:
                src_tensor = tensor_transform_reversed(src_indices)
            else:
                src_tensor = tensor_transform_normal(src_indices)
            
            tgt_tensor = tensor_transform_normal(tgt_indices)
            
            src_batch.append(src_tensor)
            tgt_batch.append(tgt_tensor)
        
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        
        src_batch = src_batch.long()
        tgt_batch = tgt_batch.long()
        
        src_batch = src_batch.t()
        tgt_batch = tgt_batch.t()
        
        return src_batch.to(self.device), tgt_batch.to(self.device)

def download_multi30k_dataset(data_dir: str = './data'):
    """Download and setup Multi30k dataset"""
    
    urls = {
        'train': "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/training.tar.gz",
        'valid': "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/validation.tar.gz"
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    for split, url in urls.items():
        split_dir = os.path.join(data_dir, split)
        print(f"Setting up {split} data...")
        
        try:
            extract_path = download_and_extract_data(url, split_dir)
            
            extracted_files = []
            for root, dirs, files in os.walk(split_dir):
                for file in files:
                    if file.endswith('.de') or file.endswith('.en'):
                        extracted_files.append(os.path.join(root, file))
            
            if extracted_files:
                print(f"  Found {len(extracted_files)} files for {split}")
                
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
            continue
    
    print("Dataset setup complete!")

def get_translation_dataloaders(data_path: str = './data', 
                               batch_size: int = 4, 
                               flip: bool = False,
                               src_lang: str = 'de',
                               tgt_lang: str = 'en'):
    """Create dataloaders without torchtext"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizers = {
        src_lang: SimpleTokenizer(src_lang),
        tgt_lang: SimpleTokenizer(tgt_lang)
    }
    
    train_dataset = Multi30kDataset(os.path.join(data_path, 'train'), src_lang, tgt_lang)
    valid_dataset = Multi30kDataset(os.path.join(data_path, 'valid'), src_lang, tgt_lang)
    
    if len(train_dataset) == 0:
        print("No training data found. Downloading...")
        download_multi30k_dataset(data_path)
        train_dataset = Multi30kDataset(os.path.join(data_path, 'train'), src_lang, tgt_lang)
        valid_dataset = Multi30kDataset(os.path.join(data_path, 'valid'), src_lang, tgt_lang)
    
    print("Building vocabularies...")
    vocabs = build_vocab(train_dataset.data, tokenizers, src_lang, tgt_lang)
    
    print(f"Source vocabulary size: {len(vocabs[src_lang])}")
    print(f"Target vocabulary size: {len(vocabs[tgt_lang])}")
    
    train_data_sorted = sorted(train_dataset.data, key=lambda x: len(tokenizers[src_lang](x[0])))
    valid_data_sorted = sorted(valid_dataset.data, key=lambda x: len(tokenizers[src_lang](x[0])))
    
    collate_fn = TranslationCollator(tokenizers, vocabs, src_lang, tgt_lang, device, flip)
    
    train_dataloader = DataLoader(train_data_sorted, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                drop_last=True,
                                shuffle=False)
    
    valid_dataloader = DataLoader(valid_data_sorted, 
                                batch_size=batch_size, 
                                collate_fn=collate_fn, 
                                drop_last=True,
                                shuffle=False)
    
    return train_dataloader, valid_dataloader, vocabs

def index_to_text(indices: torch.Tensor, vocab: SimpleVocab) -> str:
    """Convert indices back to text"""
    return " ".join([vocab.get_itos()[int(idx.item())] for idx in indices])

# Helper functions for compatibility with original code
def index_to_german(seq_de):
    return index_to_text(seq_de, vocab_transform['de'])

def index_to_eng(seq_en):
    return index_to_text(seq_en, vocab_transform['en'])

# Create a text transform object for compatibility
class TextTransform:
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab
    
    def __call__(self, text):
        tokens = self.tokenizer(text)
        indices = self.vocab(tokens)
        return tensor_transform_normal(indices)

# RNN simple example (from original code)
W_xh=torch.tensor(-10.0)
W_hh=torch.tensor(10.0)
b_h=torch.tensor(0.0)
x_t=1
h_prev=torch.tensor(-1)

X=[1,1,-1,-1,1,1]
H=[-1,-1,0,1,0,-1]

H_hat = []
t=1
for x in X:
    print("t=",t)
    x_t = x
    print("h_t-1", h_prev.item())
    h_t = torch.tanh(x_t * W_xh + h_prev * W_hh + b_h)
    h_prev = h_t
    print("x_t", x_t)
    print("h_t", h_t.item())
    print("\\n")
    H_hat.append(int(h_t.item()))
    t+=1

print("H_hat:", H_hat)
print("H:", H)

# Model architecture (unchanged from original)
class Encoder(nn.Module):
    def __init__(self, vocab_len, emb_dim, hid_dim, n_layers, dropout_prob):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_len, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        embed = self.dropout(self.embedding(input_batch))
        outputs, (hidden, cell) = self.lstm(embed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction_logit = self.fc_out(output.squeeze(0))
        prediction = self.softmax(prediction_logit)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, trg_vocab):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.trg_vocab = trg_vocab

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    train_iterator = tqdm(iterator, desc="Training", leave=False)

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_iterator.set_postfix(loss=loss.item())
        epoch_loss += loss.item()

    return epoch_loss / len(list(iterator))

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    valid_iterator = tqdm(iterator, desc="Validation", leave=False)

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)
            valid_iterator.set_postfix(loss=loss.item())
            epoch_loss += loss.item()

    return epoch_loss / len(list(iterator))

def generate_translation(model, src_sentence, src_vocab, trg_vocab, text_transform, max_len=50):
    model.eval()

    with torch.no_grad():
        src_tensor = text_transform(src_sentence).view(-1, 1).to(device)

        hidden, cell = model.encoder(src_tensor)

        trg_indexes = [trg_vocab.get_stoi()['<bos>']]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1)
        trg_tensor = trg_tensor.to(model.device)

        for _ in range(max_len):
            output, hidden, cell = model.decoder(trg_tensor[-1], hidden, cell)
            pred_token = output.argmax(1)[-1].item()
            trg_indexes.append(pred_token)

            if pred_token == trg_vocab.get_stoi()['<eos>']:
                break

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1)
            trg_tensor = trg_tensor.to(model.device)

        trg_tokens = [trg_vocab.get_itos()[i] for i in trg_indexes]

        if trg_tokens[0] == '<bos>':
            trg_tokens = trg_tokens[1:]
        if trg_tokens[-1] == '<eos>':
            trg_tokens = trg_tokens[:-1]

        translation = " ".join(trg_tokens)
        return translation

def calculate_bleu_score(generated_translation, reference_translations):
    references = [reference.split() for reference in reference_translations]
    hypothesis = generated_translation.split()
    bleu_score = sentence_bleu(references, hypothesis)
    return bleu_score

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    # Load data and create vocabularies
    print("Loading data and creating vocabularies...")
    train_dataloader, valid_dataloader, vocab_transform = get_translation_dataloaders(batch_size=4)
    
    # Create text transforms for compatibility
    tokenizers = {
        'de': SimpleTokenizer('de'),
        'en': SimpleTokenizer('en')
    }
    
    text_transform = {}
    text_transform['de'] = TextTransform(tokenizers['de'], vocab_transform['de'])
    text_transform['en'] = TextTransform(tokenizers['en'], vocab_transform['en'])
    
    # Test data loading
    print("\\nTesting data loading...")
    src, trg = next(iter(train_dataloader))
    print(f"Source batch shape: {src.shape}")
    print(f"Target batch shape: {trg.shape}")
    
    # Show some example translations
    data_itr = iter(train_dataloader)
    for n in range(1000):  # Skip to longer sequences
        try:
            german, english = next(data_itr)
        except StopIteration:
            data_itr = iter(train_dataloader)
            german, english = next(data_itr)
    
    print("\\nExample translations:")
    for n in range(3):
        try:
            german, english = next(data_itr)
            german = german.T
            english = english.T
            print("________________")
            print("German:")
            for g in german:
                print(index_to_german(g))
            print("________________")
            print("English:")
            for e in english:
                print(index_to_eng(e))
        except StopIteration:
            break
    
    # Model parameters
    INPUT_DIM = len(vocab_transform['de'])
    OUTPUT_DIM = len(vocab_transform['en'])
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 1
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    # Create model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device, trg_vocab=vocab_transform['en']).to(device)
    
    model.apply(init_weights)
    print(f'\\nThe model has {count_parameters(model):,} trainable parameters')
    
    # Setup training
    optimizer = optim.Adam(model.parameters())
    PAD_IDX = vocab_transform['en'].get_stoi()['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Load pre-trained model if available
    try:
        model.load_state_dict(torch.load('RNN-TR-model.pt', map_location=device))
        print("Loaded pre-trained model weights")
    except FileNotFoundError:
        print("No pre-trained model found. You can train the model or download weights.")
        print("To train, uncomment the training loop section in the code.")
    
    # Test translation
    print("\\n" + "="*50)
    print("TESTING TRANSLATION")
    print("="*50)
    
    # Test with example sentence
    src_sentence = 'Ein asiatischer Mann kehrt den Gehweg.'
    print(f"\\nOriginal German: {src_sentence}")
    
    try:
        generated_translation = generate_translation(
            model, 
            src_sentence=src_sentence, 
            src_vocab=vocab_transform['de'], 
            trg_vocab=vocab_transform['en'], 
            text_transform=text_transform['de'],
            max_len=12
        )
        print(f"Generated English: {generated_translation}")
        
        # Calculate BLEU score
        reference_translations = [
            "Asian man sweeping the walkway .",
            "An asian man sweeping the walkway .",
            "An Asian man sweeps the sidewalk .",
            "An Asian man is sweeping the sidewalk .",
            "An asian man is sweeping the walkway .",
            "Asian man sweeping the sidewalk ."
        ]
        
        bleu_score = calculate_bleu_score(generated_translation, reference_translations)
        print(f"BLEU Score: {bleu_score:.4f}")
        
        # Exercise 1
        print("\\n" + "-"*30)
        print("EXERCISE 1")
        print("-"*30)
        
        german_text = "Menschen gehen auf der Straße"
        english_translation = generate_translation(
            model,
            src_sentence=german_text,
            src_vocab=vocab_transform['de'],
            trg_vocab=vocab_transform['en'],
            text_transform=text_transform['de'],
            max_len=50
        )
        
        print(f"Original German text: {german_text}")
        print(f"Translated English text: {english_translation}")
        
    except Exception as e:
        print(f"Translation failed: {e}")
        print("This likely means the model needs to be trained first.")
        
    print("\\nDone! To train the model, uncomment the training loop in the code.")
