from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = "Very long document text here..."

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([text])

