# Install required packages first:
# pip install arxiv pypdf langchain-community

import arxiv
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def load_arxiv_paper(paper_id, load_full_text=True):
    """
    Load an ArXiv paper by ID.
    
    Args:
        paper_id (str): ArXiv paper ID (e.g., "1605.08386")
        load_full_text (bool): If True, downloads PDF and extracts full text.
                              If False, returns only metadata and abstract.
    
    Returns:
        list: List of Document objects
    """
    try:
        # Search for the paper
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        
        print(f"Found paper: {paper.title}")
        print(f"Authors: {[author.name for author in paper.authors]}")
        
        if not load_full_text:
            # Return just abstract and metadata
            return [Document(
                page_content=paper.summary,
                metadata={
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'arxiv_id': paper_id,
                    'published': paper.published.isoformat(),
                    'pdf_url': paper.pdf_url,
                    'categories': paper.categories,
                    'type': 'abstract'
                }
            )]
        
        # Download PDF to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Downloading PDF...")
            pdf_path = paper.download_pdf(dirpath=temp_dir)
            
            # Load PDF content
            print("Extracting text from PDF...")
            pdf_loader = PyPDFLoader(pdf_path)
            docs = pdf_loader.load()
            
            # Add ArXiv metadata to each page
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'arxiv_id': paper_id,
                    'published': paper.published.isoformat(),
                    'pdf_url': paper.pdf_url,
                    'categories': paper.categories,
                    'page_number': i + 1,
                    'total_pages': len(docs)
                })
            
            return docs
            
    except StopIteration:
        print(f"Paper {paper_id} not found")
        return []
    except Exception as e:
        print(f"Error loading paper {paper_id}: {e}")
        return []

def load_multiple_arxiv_papers(paper_ids, load_full_text=True):
    """Load multiple ArXiv papers"""
    all_docs = []
    
    for paper_id in paper_ids:
        print(f"\n--- Loading {paper_id} ---")
        docs = load_arxiv_paper(paper_id, load_full_text)
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} documents for {paper_id}")
    
    return all_docs

# Example usage:
if __name__ == "__main__":
    # Load single paper (full text)
    print("=== Loading Full Paper ===")
    docs = load_arxiv_paper("1605.08386", load_full_text=True)
    
    if docs:
        print(f"\nLoaded {len(docs)} pages")
        print(f"First page content (first 500 chars):")
        print(docs[0].page_content[:500])
        print(f"\nMetadata: {docs[0].metadata}")
    
    # Load just abstract (faster)
    print("\n=== Loading Abstract Only ===")
    abstract_docs = load_arxiv_paper("1605.08386", load_full_text=False)
    
    if abstract_docs:
        print(f"Abstract: {abstract_docs[0].page_content[:300]}...")
        print(f"Title: {abstract_docs[0].metadata['title']}")
    
    # Load multiple papers
    print("\n=== Loading Multiple Papers ===")
    paper_ids = ["1605.08386", "1706.03762"]  # Add more paper IDs as needed
    all_docs = load_multiple_arxiv_papers(paper_ids, load_full_text=False)
    print(f"Total documents loaded: {len(all_docs)}")

