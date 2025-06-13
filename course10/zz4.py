from transformers import pipeline

classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
result = classifier("Bonjour, comment ça va?")
print(result)

