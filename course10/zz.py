from transformers import pipeline
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("# Example 2: Language detection using pipeline()")
# Example 2: Language detection using pipeline()

classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
result = classifier("Bonjour, comment ça va?")
print(result)
