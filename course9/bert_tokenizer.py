from transformers import BertTokenizer

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode a sentence
encoded = tokenizer.encode_plus(
    "The quick brown fox jumps over the lazy dog.",
    add_special_tokens=True,  # [CLS] and [SEP]
    max_length=64,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'  # Return PyTorch tensors
)

print(encoded['input_ids'])       # Token IDs
print(encoded['attention_mask'])  # Mask showing which tokens are actual vs padding

