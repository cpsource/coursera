# Sequence-to-Sequence RNN Models: Translation Task

#!pip install torch==2.2.2
#!pip install torchtext==0.17.2
#!pip install portalocker==2.8.2
#!pip install torchdata==0.7.1
#!pip install pandas
#!pip install matplotlib==3.9.0 scikit-learn==1.5.0
##!pip install numpy==1.26.0
#!pip install numpy
#!pip install spacy
#!pip install nltk
#print("done")

#!python -m spacy download en_core_web_sm
#!python -m spacy download de_core_news_sm

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import sentence_bleu
import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
import random
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

W_xh=torch.tensor(-10.0)
W_hh=torch.tensor(10.0)
b_h=torch.tensor(0.0)
x_t=1
h_prev=torch.tensor(-1)

X=[1,1,-1,-1,1,1]
H=[-1,-1,0,1,0,-1]

 # Initialize an empty list to store the predicted state values
H_hat = []
# Loop through each data point in the input sequence X
t=1
for x in X:
    # Assign the current data point to x_t
    print("t=",t)
    x_t = x
    # Print the value of the previous state (h at time t-1)
    print("h_t-1", h_prev.item())

    # Compute the current state (h at time t) using the RNN formula with tanh activation
    h_t = torch.tanh(x_t * W_xh + h_prev * W_hh + b_h)

    # Update h_prev to the current state value for the next iteration
    h_prev = h_t

    # Print the current input value (x at time t)
    print("x_t", x_t)

    # Print the computed state value (h at time t)
    print("h_t", h_t.item())
    print("\n")

    # Append the current state value to the H_hat list after converting it to integer
    H_hat.append(int(h_t.item()))
    t+=1

print(H_hat)
print(H)

class Encoder(nn.Module):
    def __init__(self, vocab_len, emb_dim, hid_dim, n_layers, dropout_prob):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_len, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        #input_batch = [src len, batch size]
        embed = self.dropout(self.embedding(input_batch))
        embed = embed.to(device)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        outputs, (hidden, cell) = self.lstm(embed)

        return hidden, cell

vocab_len = 8
emb_dim = 10
hid_dim=8
n_layers=1
dropout_prob=0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_t = Encoder(vocab_len, emb_dim, hid_dim, n_layers, dropout_prob).to(device)

src_batch = torch.tensor([[0,3,4,2,1]])
# you need to transpose the input tensor as the encoder LSTM is in Sequence_first mode by default
src_batch = src_batch.t().to(device)
print("Shape of input(src) tensor:", src_batch.shape)
hidden_t , cell_t = encoder_t(src_batch)
print("Hidden tensor from encoder:",hidden_t ,"\nCell tensor from encoder:", cell_t)

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


        #input = [batch size]

        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)
        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        prediction_logit = self.fc_out(output.squeeze(0))
        prediction = self.softmax(prediction_logit)
        #prediction = [batch size, output dim]


        return prediction, hidden, cell

output_dim = 6
emb_dim=10
hid_dim = 8
n_layers=1
dropout=0.5
decoder_t = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout).to(device)

input_t = torch.tensor([0]).to(device) #<bos>
input_t.shape
prediction, hidden, cell = decoder_t(input_t, hidden_t , cell_t)
print("Prediction:", prediction, '\nHidden:',hidden,'\nCell:', cell)


#trg = [trg len, batch size]
#teacher_forcing_ratio is probability to use teacher forcing
#e.g. if teacher_forcing_ratio is 0.75 you use ground-truth inputs 75% of the time
teacher_forcing_ratio = 0.5
trg = torch.tensor([[0],[2],[3],[5],[1]]).to(device)


batch_size = trg.shape[1]
trg_len = trg.shape[0]
trg_vocab_size = decoder_t.output_dim

#tensor to store decoder outputs
outputs_t = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

#send to device

hidden_t = hidden_t.to(device)
cell_t = cell_t.to(device)


#first input to the decoder is the <bos> tokens
input = trg[0,:]


for t in range(1, trg_len):

    #you loop through the trg len and generate tokens
    #decoder receives previous generated token, cell and hidden
    # decoder outputs it prediction(probablity distribution for the next token) and updates hidden and cell
    output_t, hidden_t, cell_t = decoder_t(input, hidden_t, cell_t)

    #place predictions in a tensor holding predictions for each token
    outputs_t[t] = output_t

    #decide if you are going to use teacher forcing or not
    teacher_force = random.random() < teacher_forcing_ratio

    #get the highest predicted token from your predictions
    top1 = output_t.argmax(1)


    #if teacher forcing, use actual next token as next input
    #if not, use predicted token
    #input = trg[t] if teacher_force else top1
    input = trg[t] if teacher_force else top1

print(outputs_t,outputs_t.shape )

# Note that you need to get the argmax from the second dimension as **outputs** is an array of **output** tensors
pred_tokens = outputs_t.argmax(2)
print(pred_tokens)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device,trg_vocab):
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
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 you use ground-truth inputs 75% of the time


        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        hidden = hidden.to(device)
        cell = cell.to(device)


        #first input to the decoder is the <bos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if you are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from your predictions
            top1 = output.argmax(1)


            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            #input = trg[t] if teacher_force else top1
            input = trg[t] if teacher_force else top1


        return outputs

def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    # Wrap iterator with tqdm for progress logging
    train_iterator = tqdm(iterator, desc="Training", leave=False)

    for i, (src,trg) in enumerate(iterator):

        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)

        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)

        trg = trg[1:].contiguous().view(-1)

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        # Update tqdm progress bar with the current loss
        train_iterator.set_postfix(loss=loss.item())

        epoch_loss += loss.item()


    return epoch_loss / len(list(iterator))

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    # Wrap iterator with tqdm for progress logging
    valid_iterator = tqdm(iterator, desc="Training", leave=False)

    with torch.no_grad():

        for i, (src,trg) in enumerate(iterator):

            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)

            trg = trg[1:].contiguous().view(-1)


            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            # Update tqdm progress bar with the current loss
            valid_iterator.set_postfix(loss=loss.item())

            epoch_loss += loss.item()

    return epoch_loss / len(list(iterator))

# note, now done by la7_dataloader2.py
#wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/Multi30K_de_en_dataloader.py'
#%run Multi30K_de_en_dataloader.py

train_dataloader, valid_dataloader = get_translation_dataloaders(batch_size = 4)#,flip=True)

src, trg = next(iter(train_dataloader))
src,trg

data_itr = iter(train_dataloader)
# moving forward in the dataset to reach sequences of longer length for illustration purpose. (Remember the dataset is sorted on sequence len for optimal padding)
for n in range(1000):
    german, english= next(data_itr)

for n in range(3):
    german, english=next(data_itr)
    german=german.T
    english=english.T
    print("________________")
    print("german")
    for g in german:
        print(index_to_german(g))
    print("________________")
    print("english")
    for e in english:
        print(index_to_eng(e))

# Training the model

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

INPUT_DIM = len(vocab_transform['de'])
OUTPUT_DIM = len(vocab_transform['en'])
ENC_EMB_DIM = 128 #256
DEC_EMB_DIM = 128 #256
HID_DIM = 256 #512
N_LAYERS = 1 #2
ENC_DROPOUT = 0.3 #0.5
DEC_DROPOUT = 0.3 #0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device,trg_vocab = vocab_transform['en']).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

PAD_IDX = vocab_transform['en'].get_stoi()['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# uncomment if cuda
# torch.cuda.empty_cache()

# N_EPOCHS = 3 #run the training for at least 5 epochs
# CLIP = 1

# best_valid_loss = float('inf')
# best_train_loss = float('inf')
# train_losses = []
# valid_losses = []

# train_PPLs = []
# valid_PPLs = []

# for epoch in range(N_EPOCHS):

#     start_time = time.time()

#     train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
#     train_ppl = math.exp(train_loss)
#     valid_loss = evaluate(model, valid_dataloader, criterion)
#     valid_ppl = math.exp(valid_loss)


#     end_time = time.time()

#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)


#     if valid_loss < best_valid_loss:

#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'RNN-TR-model.pt')

#     train_losses.append(train_loss)
#     train_PPLs.append(train_ppl)
#     valid_losses.append(valid_loss)
#     valid_PPLs.append(valid_ppl)

#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')

# import matplotlib.pyplot as plt

# # Create a list of epoch numbers
# epochs = [epoch+1 for epoch in range(N_EPOCHS)]

# # Create the figure and axes
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # Plotting the training and validation loss
# ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
# ax1.plot(epochs, valid_losses, label='Validation Loss', color='orange')
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Loss')
# ax1.set_title('Training and Validation Loss/PPL')

# # Plotting the training and validation perplexity
# ax2.plot(epochs, train_PPLs, label='Train PPL', color='green')
# ax2.plot(epochs, valid_PPLs, label='Validation PPL', color='red')
# ax2.set_ylabel('Perplexity')

# # Adjust the y-axis scaling for PPL plot
# ax2.set_ylim(bottom=min(min(train_PPLs), min(valid_PPLs)) - 10, top=max(max(train_PPLs), max(valid_PPLs)) + 10)

# # Set the legend
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# lines = lines1 + lines2
# labels = labels1 + labels2
# ax1.legend(lines, labels, loc='upper right')


# # Show the plot
# plt.show()

# do this one manually from shell
#wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0201EN-Coursera/RNN-TR-model.pt'
model.load_state_dict(torch.load('RNN-TR-model.pt',map_location=torch.device('cpu')))

import torch.nn.functional as F

def generate_translation(model, src_sentence, src_vocab, trg_vocab, max_len=50):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        src_tensor = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1).to(device)

        # Pass the source tensor through the encoder
        hidden, cell = model.encoder(src_tensor)

        # Create a tensor to store the generated translation
        # get_stoi() maps tokens to indices
        trg_indexes = [trg_vocab.get_stoi()['<bos>']]  # Start with <bos> token

        # Convert the initial token to a PyTorch tensor
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1)  # Add batch dimension

        # Move the tensor to the same device as the model
        trg_tensor = trg_tensor.to(model.device)


        # Generate the translation
        for _ in range(max_len):

            # Pass the target tensor and the previous hidden and cell states through the decoder
            output, hidden, cell = model.decoder(trg_tensor[-1], hidden, cell)

            # Get the predicted next token
            pred_token = output.argmax(1)[-1].item()

            # Append the predicted token to the translation
            trg_indexes.append(pred_token)


            # If the predicted token is the <eos> token, stop generating
            if pred_token == trg_vocab.get_stoi()['<eos>']:
                break

            # Convert the predicted token to a PyTorch tensor
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1)  # Add batch dimension

            # Move the tensor to the same device as the model
            trg_tensor = trg_tensor.to(model.device)

        # Convert the generated tokens to text
        # get_itos() maps indices to tokens
        trg_tokens = [trg_vocab.get_itos()[i] for i in trg_indexes]

        # Remove the <sos> and <eos> from the translation
        if trg_tokens[0] == '<bos>':
            trg_tokens = trg_tokens[1:]
        if trg_tokens[-1] == '<eos>':
            trg_tokens = trg_tokens[:-1]

        # Return the translation list as a string

        translation = " ".join(trg_tokens)

        return translation

# model.load_state_dict(torch.load('RNN-TR-model.pt'))

# Actual translation: Asian man sweeping the walkway.
src_sentence = 'Ein asiatischer Mann kehrt den Gehweg.'


generated_translation = generate_translation(model, src_sentence=src_sentence, src_vocab=vocab_transform['de'], trg_vocab=vocab_transform['en'], max_len=12)
#generated_translation = " ".join(generated_translation_list).replace("<bos>", "").replace("<eos>", "")
print(generated_translation)

def calculate_bleu_score(generated_translation, reference_translations):
    # Convert the generated translations and reference translations into the expected format for sentence_bleu
    references = [reference.split() for reference in reference_translations]
    hypothesis = generated_translation.split()

    # Calculate the BLEU score
    bleu_score = sentence_bleu(references, hypothesis)

    return bleu_score

reference_translations = [
    "Asian man sweeping the walkway .",
    "An asian man sweeping the walkway .",
    "An Asian man sweeps the sidewalk .",
    "An Asian man is sweeping the sidewalk .",
    "An asian man is sweeping the walkway .",
    "Asian man sweeping the sidewalk ."
]

bleu_score = calculate_bleu_score(generated_translation, reference_translations)
print("BLEU Score:", bleu_score)

# Exercise 1


german_text = "Menschen gehen auf der Straße"

# The function should be defined to accept the text, the model, source and target vocabularies, and the device as parameters.
english_translation = generate_translation(
    model,
    src_sentence=german_text,
    src_vocab=vocab_transform['de'],
    trg_vocab=vocab_transform['en'],
    max_len=50
)

# Display the original and translated text
print(f"Original German text: {german_text}")
print(f"Translated English text: {english_translation}")# Define the German text to be translated


