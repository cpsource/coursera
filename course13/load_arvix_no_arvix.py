import requests
import re
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

def scrape_arxiv_metadata(paper_id):
    """
    Scrape ArXiv paper metadata from the web page (no arxiv package needed)
    """
    url = f"https://arxiv.org/abs/{paper_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        
        # Extract title
        title_match = re.search(r'<h1 class="title mathjax"[^>]*>\s*<span[^>]*>(.*?)</span>', html, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Unknown Title"
        title = re.sub(r'^Title:\s*', '', title)  # Remove "Title:" prefix
        
        # Extract authors
        authors_match = re.search(r'<div class="authors"[^>]*>(.*?)</div>', html, re.DOTALL)
        authors = []
        if authors_match:
            author_links = re.findall(r'<a[^>]*>(.*?)</a>', authors_match.group(1))
            authors = [re.sub(r'<[^>]*>', '', author).strip() for author in author_links]
        
        # Extract abstract
        abstract_match = re.search(r'<blockquote class="abstract mathjax"[^>]*>\s*<span[^>]*>Abstract:</span>\s*(.*?)</blockquote>', html, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else "No abstract found"
        abstract = re.sub(r'<[^>]*>', '', abstract)  # Remove HTML tags
        
        # Extract submission date
        date_match = re.search(r'<div class="dateline">[^(]*\(([^)]+)\)', html)
        date = date_match.group(1) if date_match else "Unknown date"
        
        # Extract categories
        subjects_match = re.search(r'<td class="tablecell subjects"[^>]*>(.*?)</td>', html, re.DOTALL)
        categories = []
        if subjects_match:
            category_spans = re.findall(r'<span[^>]*class="primary-subject"[^>]*>(.*?)</span>', subjects_match.group(1))
            categories = [cat.strip() for cat in category_spans]
        
        return {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'date': date,
            'categories': categories,
            'arxiv_id': paper_id,
            'url': url
        }
        
    except Exception as e:
        print(f"Error scraping {paper_id}: {e}")
        return None

def download_arxiv_pdf(paper_id, output_dir=None):
    """
    Download ArXiv PDF directly (no arxiv package needed)
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
    
    try:
        print(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"PDF saved to {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None

def load_arxiv_paper_web(paper_id, include_pdf=True):
    """
    Load ArXiv paper using web scraping (no arxiv package needed)
    
    Args:
        paper_id (str): ArXiv paper ID (e.g., "1605.08386")
        include_pdf (bool): Whether to download and parse the PDF
    
    Returns:
        list: List of Document objects
    """
    # Get metadata from web page
    metadata = scrape_arxiv_metadata(paper_id)
    if not metadata:
        return []
    
    documents = []
    
    if not include_pdf:
        # Return just abstract and metadata
        content = f"Title: {metadata['title']}\n\nAbstract: {metadata['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                'source': f"arxiv:{paper_id}",
                'type': 'abstract',
                **metadata
            }
        )
        return [doc]
    
    # Download and parse PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = download_arxiv_pdf(paper_id, temp_dir)
        
        if pdf_path and os.path.exists(pdf_path):
            try:
                # Load PDF with PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                
                # Add metadata to each page
                for i, doc in enumerate(docs):
                    doc.metadata.update({
                        'source': f"arxiv:{paper_id}",
                        'page_number': i + 1,
                        'total_pages': len(docs),
                        'type': 'full_text',
                        **metadata
                    })
                
                return docs
                
            except Exception as e:
                print(f"Error parsing PDF: {e}")
                # Fall back to abstract only
                content = f"Title: {metadata['title']}\n\nAbstract: {metadata['abstract']}"
                return [Document(
                    page_content=content,
                    metadata={
                        'source': f"arxiv:{paper_id}",
                        'type': 'abstract_fallback',
                        **metadata
                    }
                )]
        else:
            # PDF download failed, return abstract
            content = f"Title: {metadata['title']}\n\nAbstract: {metadata['abstract']}"
            return [Document(
                page_content=content,
                metadata={
                    'source': f"arxiv:{paper_id}",
                    'type': 'abstract_only',
                    **metadata
                }
            )]

def search_arxiv_papers(query, max_results=5):
    """
    Search ArXiv papers by query (returns paper IDs)
    """
    search_url = "https://export.arxiv.org/api/query"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        # Extract paper IDs from XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        paper_ids = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
            if id_elem is not None:
                # Extract paper ID from URL like http://arxiv.org/abs/1234.5678v1
                paper_id = id_elem.text.split('/')[-1]
                if 'v' in paper_id:
                    paper_id = paper_id.split('v')[0]  # Remove version number
                paper_ids.append(paper_id)
        
        return paper_ids
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Test with the paper you wanted
    paper_id = "1605.08386"
    
    print("=== Loading Abstract Only ===")
    docs = load_arxiv_paper_web(paper_id, include_pdf=False)
    if docs:
        print(f"Title: {docs[0].metadata['title']}")
        print(f"Authors: {docs[0].metadata['authors']}")
        print(f"Content: {docs[0].page_content[:500]}...")
    
    print("\n=== Loading Full Paper ===")
    docs = load_arxiv_paper_web(paper_id, include_pdf=True)
    if docs:
        print(f"Loaded {len(docs)} pages")
        print(f"First page: {docs[0].page_content[:300]}...")
    
    print("\n=== Search Example ===")
    paper_ids = search_arxiv_papers("attention mechanism", max_results=3)
    print(f"Found papers: {paper_ids}")

