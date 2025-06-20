The **LangChain `Document` object** is a lightweight, structured way to represent a piece of **text** along with its **metadata**, used widely in LangChain pipelines for RAG (Retrieval-Augmented Generation), chunking, indexing, and retrieval.

---

### 🔹 Core Purpose

The `Document` object allows you to:

* Keep a **text chunk** together with its **source metadata**
* Process documents uniformly in vectorstores, retrievers, agents, etc.
* Support traceability (e.g., where a chunk came from in a larger doc)

---

### 🔹 Import & Structure

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the text of the document.",
    metadata={"source": "file1.txt", "page": 1}
)
```

---

### 🔹 Main Attributes

| Attribute      | Type   | Description                                         |
| -------------- | ------ | --------------------------------------------------- |
| `page_content` | `str`  | The actual text content of the document             |
| `metadata`     | `dict` | Optional metadata (e.g. filename, URL, page number) |

---

### 🔹 Example Use Case (Chunking + Vectorstore)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.create_documents(["Full document text here."], metadata={"source": "report.pdf"})

# Each `doc` in `docs` is a Document object:
print(docs[0].page_content)
print(docs[0].metadata)
```

---

### 🔹 Usage in RAG

* Ingested documents are split into `Document` chunks.
* These chunks are embedded and stored in a vectorstore (e.g., FAISS).
* When a query is issued, top-k `Document` objects are retrieved and used to generate answers.

---

### 🔹 Summary

The `Document` object in LangChain is:

* 📄 A simple text + metadata container
* 🔧 Used in all major LangChain modules: retrievers, loaders, splitters, and agents
* 🧩 Essential for structuring data in LLM workflows

Let me know if you'd like examples for loading PDFs, web pages, or databases into `Document` format.


