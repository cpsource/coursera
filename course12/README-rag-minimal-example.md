Here is a **minimal RAG (Retriever-Augmented Generation)** implementation in Python using Hugging Face and FAISS. This version includes **extensive inline documentation** for each step so you can learn while reading.

---

### ✅ Features:

* Loads a few documents.
* Uses **DPR for retrieval**.
* Uses **GPT-2 for generation**.
* Retrieves relevant contexts from FAISS.
* Generates an answer with and without context for comparison.

---

### 🔧 Prerequisites (run first):

```bash
pip install transformers faiss-cpu torch
```

---

### 🧠 Minimal RAG Implementation

```python
import torch
import numpy as np
import faiss
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForCausalLM
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------------
# STEP 1: Prepare a small corpus of documents
# -----------------------------------------------------------------------------------
documents = [
    "The company offers 15 days of paid vacation per year.",
    "Employees should submit reimbursement forms within 30 days.",
    "Mobile phones must be secured with a company-approved password.",
    "Remote work is allowed up to 3 days per week.",
    "Drinking alcohol during work hours is strictly prohibited."
]

# -----------------------------------------------------------------------------------
# STEP 2: Load DPR Context Encoder and Tokenizer
# These will convert documents into dense embeddings
# -----------------------------------------------------------------------------------
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Tokenize and encode each document
context_embeddings = []
for doc in documents:
    inputs = context_tokenizer(doc, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        embedding = context_encoder(**inputs).pooler_output  # (1, 768)
    context_embeddings.append(embedding.cpu().numpy())

# Stack into numpy array for FAISS
context_embeddings_np = np.vstack(context_embeddings).astype('float32')  # Shape: (5, 768)

# -----------------------------------------------------------------------------------
# STEP 3: Create FAISS index
# This allows us to retrieve similar documents by vector similarity
# -----------------------------------------------------------------------------------
embedding_dim = context_embeddings_np.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 = Euclidean distance
index.add(context_embeddings_np)  # Add document embeddings to index

# -----------------------------------------------------------------------------------
# STEP 4: Load DPR Question Encoder
# This will embed the user query in the same vector space
# -----------------------------------------------------------------------------------
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# -----------------------------------------------------------------------------------
# STEP 5: Define a query and retrieve top-k relevant documents
# -----------------------------------------------------------------------------------
query = "What is the mobile phone policy?"

# Tokenize and encode the question
inputs = question_tokenizer(query, return_tensors="pt").to(device)
with torch.no_grad():
    query_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()

# Search FAISS for top 2 closest documents
D, I = index.search(query_embedding, k=2)

print("Top matching documents:")
for idx in I[0]:
    print("-", documents[idx])

# -----------------------------------------------------------------------------------
# STEP 6: Load GPT-2 for generation
# -----------------------------------------------------------------------------------
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
gpt_model.eval()

# Set special token to avoid warnings
gpt_model.generation_config.pad_token_id = gpt_tokenizer.eos_token_id

# -----------------------------------------------------------------------------------
# STEP 7a: Generate answer WITHOUT context
# -----------------------------------------------------------------------------------
def generate_without_context(query):
    inputs = gpt_tokenizer(query, return_tensors="pt").to(device)
    output = gpt_model.generate(inputs["input_ids"], max_new_tokens=50)
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)

print("\nAnswer without context:")
print(generate_without_context(query))

# -----------------------------------------------------------------------------------
# STEP 7b: Generate answer WITH top-k context
# -----------------------------------------------------------------------------------
def generate_with_context(query, retrieved_docs):
    full_input = query + " " + " ".join(retrieved_docs)
    inputs = gpt_tokenizer(full_input, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output = gpt_model.generate(inputs["input_ids"], max_new_tokens=50)
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)

# Use the top 2 FAISS matches
retrieved = [documents[i] for i in I[0]]

print("\nAnswer with retrieved context:")
print(generate_with_context(query, retrieved))
```

---

### 🔍 Sample Output (varies with random gen):

```
Top matching documents:
- Mobile phones must be secured with a company-approved password.
- The company offers 15 days of paid vacation per year.

Answer without context:
The mobile phone policy is a very important part of your job. You should...

Answer with retrieved context:
Mobile phones must be secured with a company-approved password. The mobile...
```

---

## 🔄 Recap

| Step | Component    | Role                                 |
| ---- | ------------ | ------------------------------------ |
| 1    | Documents    | Input corpus                         |
| 2    | DPR Context  | Convert docs → dense vectors         |
| 3    | FAISS        | Search by similarity (top-k matches) |
| 4    | DPR Question | Encode query in same space           |
| 5    | GPT-2        | Generate answer from query + docs    |

---

Would you like this turned into a class or saved as a reusable module for RAG experiments?


