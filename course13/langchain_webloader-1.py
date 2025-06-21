#!/usr/bin/env python3
"""
LangChain WebBaseLoader Program

This program demonstrates how to use LangChain's WebBaseLoader to fetch and display web pages.
It's like having a specialized librarian that not only reads web pages but also prepares them
for AI processing by converting them into LangChain Document objects.

Usage:
    python langchain_webloader.py <URL>
    python langchain_webloader.py <URL> --format content
    python langchain_webloader.py <URL> --format metadata
    python langchain_webloader.py <URL> --format both
    python langchain_webloader.py <URL> --format links
    python langchain_webloader.py <URL> --format headings

Requirements:
    pip install langchain-community beautifulsoup4 lxml
"""

import os
# Set USER_AGENT before importing langchain to avoid warning
if 'USER_AGENT' not in os.environ:
    os.environ['USER_AGENT'] = "LangChainWebLoader/1.0 (Educational Purpose)"

import sys
import argparse
from langchain_community.document_loaders import WebBaseLoader
from urllib.parse import urlparse, urljoin
import json
import re
from bs4 import BeautifulSoup

class LangChainWebLoader:
    def __init__(self, timeout=10):
        """Initialize the LangChain web loader"""
        self.timeout = timeout
    
    def load_webpage(self, url):
        """
        Load a webpage using LangChain's WebBaseLoader
        Think of this as having a smart assistant that reads web pages
        and converts them into a format that AI can easily understand
        """
        try:
            # Ensure URL has a scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            print(f"Loading webpage: {url}")
            
            # Create the WebBaseLoader with custom configuration
            loader = WebBaseLoader(
                url,
                requests_kwargs={
                    'timeout': self.timeout,
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                }
            )
            
            # Load the document(s) - this returns a list of Document objects
            documents = loader.load()
            
            print(f"Successfully loaded {len(documents)} document(s)")
            return documents
            
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return None
    
    def load_multiple_webpages(self, urls):
        """Load multiple webpages at once"""
        try:
            # Ensure all URLs have schemes
            processed_urls = []
            for url in urls:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                processed_urls.append(url)
            
            print(f"Loading {len(processed_urls)} webpages...")
            
            # WebBaseLoader can handle multiple URLs
            loader = WebBaseLoader(processed_urls)
            documents = loader.load()
            
            print(f"Successfully loaded {len(documents)} document(s) from {len(processed_urls)} URLs")
            return documents
            
        except Exception as e:
            print(f"Error loading multiple URLs: {e}")
            return None
    
    def get_soup_from_loader(self, url):
        """
        Get the raw HTML and create BeautifulSoup object for advanced parsing
        This bridges LangChain and Beautiful Soup capabilities
        """
        try:
            import requests
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup, response.url
            
        except Exception as e:
            print(f"Error getting raw HTML: {e}")
            return None, None
    
    def display_content_preview(self, documents):
        """Display a meaningful preview of content, similar to simple_demo"""
        if not documents:
            return
        
        content = documents[0].page_content
        
        # Try to find the main content by skipping navigation/header stuff
        lines = content.split('\n')
        meaningful_lines = []
        
        # Skip common header/nav patterns and find substantial content
        for line in lines:
            line = line.strip()
            if len(line) > 50 and not line.startswith(('Skip to', 'Menu', 'Search', 'Products', 'Solutions', 'Sign in', 'Contact')):
                meaningful_lines.append(line)
                if len(' '.join(meaningful_lines)) > 800:  # Get about 800 chars of good content
                    break
        
        if meaningful_lines:
            display_content = ' '.join(meaningful_lines)
        else:
            # Fallback to original method if we can't find good content
            display_content = content[:1000]
        
        print("\nContent preview:")
        print("-" * 50)
        print(display_content[:1000])
        
        if len(content) > 1000:
            print(f"\n... [showing first portion of {len(content):,} total characters]")
    
    def display_pretty(self, documents):
        """Display the document content in a pretty format"""
        print("\n" + "="*60)
        print("LANGCHAIN DOCUMENT CONTENT (PRETTY FORMAT)")
        print("="*60)
        
        for i, doc in enumerate(documents, 1):
            if len(documents) > 1:
                print(f"\n--- Document {i} ---")
            
            # Display the cleaned content that LangChain provides
            content = doc.page_content
            
            # Format it nicely with line breaks
            lines = content.split('\n')
            for line in lines[:50]:  # Show first 50 lines
                if line.strip():
                    print(line)
            
            if len(lines) > 50:
                print(f"\n... [Content truncated. Total lines: {len(lines)}]")
    
    def display_text_only(self, documents):
        """Display clean text content - LangChain already does this cleaning"""
        print("\n" + "="*60)
        print("TEXT CONTENT (CLEANED BY LANGCHAIN)")
        print("="*60)
        
        for i, doc in enumerate(documents, 1):
            if len(documents) > 1:
                print(f"\n--- Document {i} ---")
            
            content = doc.page_content
            
            # LangChain already cleaned the text, so we just display it
            if len(content) > 2000:
                print(content[:2000])
                print(f"\n... [Content truncated. Total length: {len(content)} characters]")
            else:
                print(content)
    
    def display_title_and_meta(self, documents):
        """Display document metadata from LangChain"""
        print("\n" + "="*60)
        print("LANGCHAIN DOCUMENT METADATA")
        print("="*60)
        
        for i, doc in enumerate(documents, 1):
            if len(documents) > 1:
                print(f"\n--- Document {i} Metadata ---")
            
            metadata = doc.metadata
            
            # Display all metadata
            for key, value in metadata.items():
                print(f"{key}: {value}")
            
            # Additional statistics
            content_length = len(doc.page_content)
            word_count = len(doc.page_content.split())
            line_count = len(doc.page_content.splitlines())
            
            print(f"\nContent Statistics:")
            print(f"  Characters: {content_length:,}")
            print(f"  Words: {word_count:,}")
            print(f"  Lines: {line_count:,}")
    
    def display_links(self, documents):
        """Extract and display links using Beautiful Soup on the source URL"""
        print("\n" + "="*60)
        print("LINKS FOUND ON PAGE")
        print("="*60)
        
        # Get the source URL from the first document
        if not documents:
            print("No documents to analyze")
            return
        
        source_url = documents[0].metadata.get('source')
        if not source_url:
            print("No source URL found in metadata")
            return
        
        # Use Beautiful Soup to get links
        soup, base_url = self.get_soup_from_loader(source_url)
        if not soup:
            print("Could not parse HTML for links")
            return
        
        links = soup.find_all('a', href=True)
        
        if not links:
            print("No links found")
            return
        
        print(f"Found {len(links)} links:")
        print("-" * 30)
        
        for i, link in enumerate(links[:20], 1):  # Show first 20 links
            href = link['href']
            text = link.get_text().strip()
            
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Clean up text display
            if not text:
                text = "[No text]"
            elif len(text) > 60:
                text = text[:57] + "..."
            
            print(f"{i:3d}. {text}")
            print(f"     URL: {full_url}")
        
        if len(links) > 20:
            print(f"\n... and {len(links) - 20} more links")
    
    def display_headings(self, documents):
        """Extract and display headings using Beautiful Soup"""
        print("\n" + "="*60)
        print("PAGE HEADINGS (TABLE OF CONTENTS)")
        print("="*60)
        
        if not documents:
            print("No documents to analyze")
            return
        
        source_url = documents[0].metadata.get('source')
        if not source_url:
            print("No source URL found in metadata")
            return
        
        # Use Beautiful Soup to get headings
        soup, _ = self.get_soup_from_loader(source_url)
        if not soup:
            print("Could not parse HTML for headings")
            return
        
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if not headings:
            print("No headings found")
            return
        
        for heading in headings:
            level = heading.name
            text = heading.get_text().strip()
            indent = "  " * (int(level[1]) - 1)  # Indent based on heading level
            print(f"{indent}{level.upper()}: {text}")
    
    def display_images(self, documents):
        """Extract and display images using Beautiful Soup"""
        print("\n" + "="*60)
        print("IMAGES FOUND ON PAGE")
        print("="*60)
        
        if not documents:
            print("No documents to analyze")
            return
        
        source_url = documents[0].metadata.get('source')
        if not source_url:
            print("No source URL found in metadata")
            return
        
        # Use Beautiful Soup to get images
        soup, base_url = self.get_soup_from_loader(source_url)
        if not soup:
            print("Could not parse HTML for images")
            return
        
        images = soup.find_all('img')
        
        if not images:
            print("No images found")
            return
        
        print(f"Found {len(images)} images:")
        print("-" * 30)
        
        for i, img in enumerate(images[:15], 1):  # Show first 15 images
            src = img.get('src', 'No source')
            alt = img.get('alt', 'No alt text')
            
            # Convert relative URLs to absolute
            if src != 'No source':
                full_url = urljoin(base_url, src)
            else:
                full_url = 'No source'
            
            print(f"{i:3d}. Alt text: {alt}")
            print(f"     Source: {full_url}")
        
        if len(images) > 15:
            print(f"\n... and {len(images) - 15} more images")

def main():
    parser = argparse.ArgumentParser(
        description="Load and display web pages using LangChain's WebBaseLoader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python langchain_webloader.py https://example.com
    python langchain_webloader.py example.com --format text
    python langchain_webloader.py https://python.org --format metadata
    python langchain_webloader.py https://news.ycombinator.com --format links
    python langchain_webloader.py https://wikipedia.org --format headings
    python langchain_webloader.py https://example.com --format images
        """
    )
    
    parser.add_argument('url', help='URL to fetch and display')
    parser.add_argument(
        '--format', '-f',
        choices=['pretty', 'text', 'metadata', 'links', 'headings', 'images', 'all'],
        default='all',
        help='Display format (default: all)'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=10,
        help='Request timeout in seconds (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create the LangChain web loader - like getting a smart web browser
    loader = LangChainWebLoader(timeout=args.timeout)
    
    # Load the webpage
    documents = loader.load_webpage(args.url)
    
    if not documents:
        print("Failed to load the page")
        sys.exit(1)
    
    # Show basic info about what we loaded
    doc = documents[0]
    print(f"Successfully loaded: {doc.metadata.get('source', args.url)}")
    print(f"Content length: {len(doc.page_content):,} characters")
    print(f"Word count: {len(doc.page_content.split()):,} words")
    
    # Show content preview using the same logic as simple_demo
    loader.display_content_preview(documents)
    
    # Display based on requested format
    if args.format == 'pretty' or args.format == 'all':
        loader.display_pretty(documents)
    
    if args.format == 'text' or args.format == 'all':
        loader.display_text_only(documents)
    
    if args.format == 'metadata' or args.format == 'all':
        loader.display_title_and_meta(documents)
    
    if args.format == 'headings' or args.format == 'all':
        loader.display_headings(documents)
    
    if args.format == 'links' or args.format == 'all':
        loader.display_links(documents)
    
    if args.format == 'images' or args.format == 'all':
        loader.display_images(documents)

def simple_demo():
    """
    Simple demo function - exactly like your image example
    """
    print("Simple LangChain WebBaseLoader Demo")
    print("=" * 40)
    
    # Replicate the exact example from your image
    from langchain_community.document_loaders import WebBaseLoader
    
    # Use the same URL from your example
    loader = WebBaseLoader("https://www.ibm.com/topics/langchain")
    
    data = loader.load()
    
    # Display exactly like your example would
    print(f"Loaded {len(data)} document(s)")
    
    # Get the content and clean it up for better display
    content = data[0].page_content
    
    # Try to find the main content by skipping navigation/header stuff
    lines = content.split('\n')
    meaningful_lines = []
    
    # Skip common header/nav patterns and find substantial content
    for line in lines:
        line = line.strip()
        if len(line) > 50 and not line.startswith(('Skip to', 'Menu', 'Search', 'IBM', 'Products', 'Solutions')):
            meaningful_lines.append(line)
            if len(' '.join(meaningful_lines)) > 800:  # Get about 800 chars of good content
                break
    
    if meaningful_lines:
        display_content = ' '.join(meaningful_lines)
    else:
        # Fallback to original method if we can't find good content
        display_content = content[:1000]
    
    print("\nMain content preview:")
    print("-" * 50)
    print(display_content[:1000])
    
    if len(content) > 1000:
        print(f"\n... [showing first portion of {len(content):,} total characters]")
    
    print(f"\nDocument metadata:")
    print(f"Source: {data[0].metadata.get('source', 'Unknown')}")
    print(f"Total content length: {len(content):,} characters")
    print(f"Word count: {len(content.split()):,} words")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No command line arguments - run simple demo like your image
        simple_demo()
    else:
        # Command line arguments provided - run full program
        main()
