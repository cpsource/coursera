# Fixed BERT Pre-training Code without TorchText Dependencies

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import random
from itertools import chain
import pandas as pd
from copy import deepcopy
import csv
import json
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

# Device setup with proper fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
PAD_IDX = 0
EMBEDDING_DIM = 10
VOCAB_SIZE = 147161

class BERTCSVDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            bert_input = torch.tensor(json.loads(row['BERT Input']), dtype=torch.long)
            bert_label = torch.tensor(json.loads(row['BERT Label']), dtype=torch.long)
            segment_label = torch.tensor([int(x) for x in row['Segment Label'].split(',')], dtype=torch.long)
            is_next = torch.tensor(row['Is Next'], dtype=torch.long)
            original_text = row['Original Text']
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for row {idx}: {e}")
            print("BERT Input:", row['BERT Input'])
            print("BERT Label:", row['BERT Label'])
            return None
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return None

        # Tokenizing with BERT
        encoded_input = self.tokenizer.encode_plus(
            original_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()

        return (bert_input, bert_label, segment_label, is_next, input_ids, attention_mask, original_text)

def collate_batch(batch):
    """Collate function that handles None values and pads sequences"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    bert_inputs_batch, bert_labels_batch, segment_labels_batch = [], [], []
    is_nexts_batch, input_ids_batch, attention_mask_batch, original_text_batch = [], [], [], []

    for bert_input, bert_label, segment_label, is_next, input_ids, attention_mask, original_text in batch:
        bert_inputs_batch.append(bert_input)
        bert_labels_batch.append(bert_label)
        segment_labels_batch.append(segment_label)
        is_nexts_batch.append(is_next)
        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        original_text_batch.append(original_text)

    # Pad sequences
    bert_inputs_final = pad_sequence(bert_inputs_batch, padding_value=PAD_IDX, batch_first=False)
    bert_labels_final = pad_sequence(bert_labels_batch, padding_value=PAD_IDX, batch_first=False)
    segment_labels_final = pad_sequence(segment_labels_batch, padding_value=PAD_IDX, batch_first=False)
    is_nexts_batch = torch.stack(is_nexts_batch)

    return bert_inputs_final, bert_labels_final, segment_labels_final, is_nexts_batch

class TokenEmbedding(nn.Module):
    """Token embedding layer that converts token IDs to dense vectors"""
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    """Adds positional information to token embeddings"""
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class BERTEmbedding(nn.Module):
    """Combines token, positional, and segment embeddings"""
    def __init__(self, vocab_size, emb_size, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        self.segment_embedding = nn.Embedding(3, emb_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, bert_inputs, segment_labels=None):
        token_emb = self.token_embedding(bert_inputs)
        pos_emb = self.positional_encoding(token_emb)
        
        if segment_labels is not None:
            seg_emb = self.segment_embedding(segment_labels)
            x = self.dropout(token_emb + pos_emb + seg_emb)
        else:
            x = self.dropout(token_emb + pos_emb)
        
        return x

class BERT(torch.nn.Module):
    """
    BERT model for pre-training with MLM and NSP tasks
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # Embedding layer that combines token, positional, and segment embeddings
        self.bert_embedding = BERTEmbedding(vocab_size, d_model, dropout)

        # Transformer Encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=heads, 
            dropout=dropout, 
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        # Output layers
        self.nextsentenceprediction = nn.Linear(d_model, 2)
        self.masked_language = nn.Linear(d_model, vocab_size)

    def forward(self, bert_inputs, segment_labels):
        # Create padding mask - ensures attention doesn't focus on padding tokens
        padding_mask = (bert_inputs == PAD_IDX).transpose(0, 1).to(bert_inputs.device)
        
        # Generate combined embeddings
        bert_embedding = self.bert_embedding(bert_inputs, segment_labels)
        
        # Pass through transformer encoder stack
        transformer_output = self.transformer_encoder(
            bert_embedding, 
            src_key_padding_mask=padding_mask
        )
        
        # NSP prediction from [CLS] token (first token)
        next_sentence_prediction = self.nextsentenceprediction(transformer_output[0, :])
        
        # MLM prediction for all tokens
        masked_language = self.masked_language(transformer_output)
        
        return next_sentence_prediction, masked_language

def evaluate(dataloader, model, loss_fn_mlm, loss_fn_nsp, device):
    """Evaluate model performance on given dataloader"""
    model.eval()
    total_loss = 0
    total_next_sentence_loss = 0
    total_mask_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
                
            bert_inputs, bert_labels, segment_labels, is_nexts = batch
            
            # Move all tensors to the same device
            bert_inputs = bert_inputs.to(device)
            bert_labels = bert_labels.to(device)
            segment_labels = segment_labels.to(device)
            is_nexts = is_nexts.to(device)

            # Forward pass
            next_sentence_prediction, masked_language = model(bert_inputs, segment_labels)

            # Calculate losses
            next_loss = loss_fn_nsp(next_sentence_prediction, is_nexts)
            mask_loss = loss_fn_mlm(
                masked_language.view(-1, masked_language.size(-1)), 
                bert_labels.view(-1)
            )

            loss = next_loss + mask_loss
            
            # Only accumulate valid losses (not NaN)
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_next_sentence_loss += next_loss.item()
                total_mask_loss += mask_loss.item()
                total_batches += 1

    if total_batches == 0:
        print("No valid batches found during evaluation")
        return 0.0
        
    avg_loss = total_loss / total_batches
    avg_next_sentence_loss = total_next_sentence_loss / total_batches
    avg_mask_loss = total_mask_loss / total_batches

    print(f"Average Loss: {avg_loss:.4f}, Average Next Sentence Loss: {avg_next_sentence_loss:.4f}, Average Mask Loss: {avg_mask_loss:.4f}")
    return avg_loss

def create_sample_data(num_samples=100):
    """Create sample training data if real data is not available"""
    print("Creating sample data for testing...")
    
    sample_sentences = [
        "The cat sat on the mat.",
        "It was a sunny day outside.",
        "Machine learning is fascinating.",
        "Natural language processing helps computers understand text.",
        "BERT is a transformer model.",
        "Deep learning requires lots of data.",
        "Python is a popular programming language.",
        "Artificial intelligence is changing the world.",
        "Data science combines statistics and programming.",
        "Neural networks can learn complex patterns."
    ]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data = []
    
    for i in range(num_samples):
        # Create random sentence pairs
        sent1 = random.choice(sample_sentences)
        sent2 = random.choice(sample_sentences)
        
        # Randomly decide if they're consecutive (for NSP task)
        is_next = random.choice([0, 1])
        
        # Tokenize the pair
        encoded = tokenizer.encode_plus(
            sent1, sent2,
            add_special_tokens=True,
            max_length=50,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].squeeze().tolist()
        
        # Create labels for MLM (randomly mask 15% of tokens)
        labels = [-100] * len(input_ids)  # -100 is ignored by CrossEntropyLoss
        segment_labels = encoded['token_type_ids'].squeeze().tolist()
        
        # Randomly mask some tokens for MLM
        for j in range(len(input_ids)):
            if random.random() < 0.15 and input_ids[j] not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                labels[j] = input_ids[j]
                input_ids[j] = tokenizer.mask_token_id
        
        data.append({
            'BERT Input': json.dumps(input_ids),
            'BERT Label': json.dumps(labels),
            'Segment Label': ','.join(map(str, segment_labels)),
            'Is Next': is_next,
            'Original Text': f"{sent1} [SEP] {sent2}"
        })
    
    return pd.DataFrame(data)

def train_model_with_sample_data():
    """Train model with sample data for demonstration"""
    # Model parameters - ensure heads divides d_model evenly
    d_model = EMBEDDING_DIM
    n_layers = 2
    heads = 2  # Changed to 2 to divide evenly into d_model=10
    dropout = 0.1

    # Create model
    model = BERT(VOCAB_SIZE, d_model, n_layers, heads, dropout)
    model.to(device)
    print(f"Model moved to {device}")

    # Loss functions
    loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 to ignore non-masked tokens
    loss_fn_nsp = nn.CrossEntropyLoss()

    # Create sample datasets
    print("Creating sample datasets...")
    train_df = create_sample_data(200)
    test_df = create_sample_data(50)
    
    # Save sample data to CSV for testing
    train_df.to_csv('sample_train_data.csv', index=False)
    test_df.to_csv('sample_test_data.csv', index=False)
    
    # Create datasets and dataloaders
    class SampleDataset(Dataset):
        def __init__(self, df):
            self.data = df
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            try:
                bert_input = torch.tensor(json.loads(row['BERT Input']), dtype=torch.long)
                bert_label = torch.tensor(json.loads(row['BERT Label']), dtype=torch.long)
                segment_label = torch.tensor([int(x) for x in row['Segment Label'].split(',')], dtype=torch.long)
                is_next = torch.tensor(row['Is Next'], dtype=torch.long)
                original_text = row['Original Text']
                return (bert_input, bert_label, segment_label, is_next, None, None, original_text)
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                return None

    train_dataset = SampleDataset(train_df)
    test_dataset = SampleDataset(test_df)
    
    batch_size = 3
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    
    num_epochs = 20
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # Training loop
    train_losses = []
    eval_losses = []

    print("Starting training...")
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        total_loss = 0
        valid_batches = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            if batch is None:
                continue
                
            bert_inputs, bert_labels, segment_labels, is_nexts = batch
            
            # Move to device
            bert_inputs = bert_inputs.to(device)
            bert_labels = bert_labels.to(device)
            segment_labels = segment_labels.to(device)
            is_nexts = is_nexts.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            next_sentence_prediction, masked_language = model(bert_inputs, segment_labels)

            # Calculate losses
            next_loss = loss_fn_nsp(next_sentence_prediction, is_nexts)
            mask_loss = loss_fn_mlm(
                masked_language.view(-1, masked_language.size(-1)), 
                bert_labels.view(-1)
            )

            loss = next_loss + mask_loss
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                valid_batches += 1

        if valid_batches > 0:
            avg_train_loss = total_loss / valid_batches
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")

            # Evaluation after each epoch
            eval_loss = evaluate(test_dataloader, model, loss_fn_mlm, loss_fn_nsp, device)
            eval_losses.append(eval_loss)
        else:
            print(f"Epoch {epoch+1} - No valid batches found")

    # Plot losses if we have data
    if train_losses and eval_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker='o')
        plt.plot(range(1, len(eval_losses)+1), eval_losses, label="Evaluation Loss", marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    return model

# Inference functions with proper device handling
def predict_nsp(sentence1, sentence2, model, tokenizer, device):
    """Predict if second sentence follows the first"""
    model.eval()
    
    # Tokenize
    tokens = tokenizer.encode_plus(sentence1, sentence2, return_tensors="pt")
    tokens_tensor = tokens["input_ids"].to(device)
    segment_tensor = tokens["token_type_ids"].to(device)

    with torch.no_grad():
        nsp_prediction, _ = model(tokens_tensor, segment_tensor)
        first_logits = nsp_prediction[0].unsqueeze(0)
        logits = torch.softmax(first_logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()

    return "Second sentence follows the first" if prediction == 1 else "Second sentence does not follow the first"

def predict_mlm(sentence, model, tokenizer, device):
    """Predict masked tokens in a sentence"""
    model.eval()
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens_tensor = inputs.input_ids.to(device)
    segment_labels = torch.zeros_like(tokens_tensor).to(device)

    with torch.no_grad():
        output_tuple = model(tokens_tensor, segment_labels)
        predictions = output_tuple[1]

        # Find mask token
        mask_token_index = (tokens_tensor == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_token_index) > 0:
            predicted_index = torch.argmax(predictions[0, mask_token_index.item(), :], dim=-1)
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index.item()])[0]
            predicted_sentence = sentence.replace(tokenizer.mask_token, predicted_token, 1)
        else:
            predicted_sentence = sentence

    return predicted_sentence

# Main execution
if __name__ == "__main__":
    print("BERT Pre-training Code (TorchText-free version)")
    print("=" * 50)
    
    # Train model with sample data
    model = train_model_with_sample_data()
    
    # Initialize tokenizer for inference
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("\nTesting inference functions:")
    print("=" * 30)
    
    # Test NSP
    sentence1 = "The cat sat on the mat."
    sentence2 = "It was a sunny day"
    nsp_result = predict_nsp(sentence1, sentence2, model, tokenizer, device)
    print(f"NSP Result: {nsp_result}")
    
    # Test MLM
    masked_sentence = "The cat sat on the [MASK]."
    mlm_result = predict_mlm(masked_sentence, model, tokenizer, device)
    print(f"MLM Result: {mlm_result}")
    
    print("\nTraining completed successfully!")


