#!/usr/bin/env python3
"""
Beautiful Soup Web Page Loader

This program demonstrates how to use Beautiful Soup to fetch and display web pages.
It accepts a URL from the command line, fetches the content, and displays it in various formats.

Usage:
    python webpage_loader.py <URL>
    python webpage_loader.py <URL> --format text
    python webpage_loader.py <URL> --format pretty
    python webpage_loader.py <URL> --format links
    python webpage_loader.py <URL> --format title

Requirements:
    pip install requests beautifulsoup4 lxml
"""

import sys
import argparse
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

class WebPageLoader:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.session = requests.Session()
        # Set a user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_page(self, url):
        """
        Fetch a web page and return a BeautifulSoup object
        Think of this as opening a book and preparing to read it
        """
        try:
            # Ensure URL has a scheme (like adding 'http://' if missing)
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Create soup object - like getting a smart reader for the book
            soup = BeautifulSoup(response.content, 'html.parser')
            
            return soup, response
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None, None
    
    def display_pretty(self, soup):
        """Display the HTML in a nicely formatted way"""
        print("\n" + "="*50)
        print("PRETTY FORMATTED HTML")
        print("="*50)
        print(soup.prettify())
    
    def display_text_only(self, soup):
        """Extract and display only the text content (no HTML tags)"""
        print("\n" + "="*50)
        print("TEXT CONTENT ONLY")
        print("="*50)
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace - like removing extra spaces from a document
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        print(text)
    
    def display_title_and_meta(self, soup):
        """Display page title and meta information"""
        print("\n" + "="*50)
        print("PAGE INFORMATION")
        print("="*50)
        
        # Title - like the cover of a book
        title = soup.find('title')
        if title:
            print(f"Title: {title.get_text().strip()}")
        else:
            print("Title: Not found")
        
        # Meta description - like a book's summary
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            print(f"Description: {meta_desc.get('content', 'No content')}")
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            print(f"Keywords: {meta_keywords.get('content', 'No content')}")
        
        # Author
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author:
            print(f"Author: {meta_author.get('content', 'No content')}")
    
    def display_links(self, soup, base_url):
        """Display all links found on the page"""
        print("\n" + "="*50)
        print("LINKS FOUND ON PAGE")
        print("="*50)
        
        links = soup.find_all('a', href=True)
        
        if not links:
            print("No links found")
            return
        
        print(f"Found {len(links)} links:")
        print("-" * 30)
        
        for i, link in enumerate(links, 1):
            href = link['href']
            text = link.get_text().strip()
            
            # Convert relative URLs to absolute - like adding full address to partial ones
            full_url = urljoin(base_url, href)
            
            # Clean up text display
            if not text:
                text = "[No text]"
            elif len(text) > 60:
                text = text[:57] + "..."
            
            print(f"{i:3d}. {text}")
            print(f"     URL: {full_url}")
            print()
    
    def display_headings(self, soup):
        """Display all headings (h1, h2, h3, etc.) - like a table of contents"""
        print("\n" + "="*50)
        print("PAGE HEADINGS (TABLE OF CONTENTS)")
        print("="*50)
        
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if not headings:
            print("No headings found")
            return
        
        for heading in headings:
            level = heading.name
            text = heading.get_text().strip()
            indent = "  " * (int(level[1]) - 1)  # Indent based on heading level
            print(f"{indent}{level.upper()}: {text}")
    
    def display_images(self, soup, base_url):
        """Display information about images on the page"""
        print("\n" + "="*50)
        print("IMAGES FOUND ON PAGE")
        print("="*50)
        
        images = soup.find_all('img')
        
        if not images:
            print("No images found")
            return
        
        print(f"Found {len(images)} images:")
        print("-" * 30)
        
        for i, img in enumerate(images, 1):
            src = img.get('src', 'No source')
            alt = img.get('alt', 'No alt text')
            
            # Convert relative URLs to absolute
            if src != 'No source':
                full_url = urljoin(base_url, src)
            else:
                full_url = 'No source'
            
            print(f"{i:3d}. Alt text: {alt}")
            print(f"     Source: {full_url}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description="Load and display web pages using Beautiful Soup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python webpage_loader.py https://example.com
    python webpage_loader.py example.com --format text
    python webpage_loader.py https://news.ycombinator.com --format links
    python webpage_loader.py https://python.org --format title
        """
    )
    
    parser.add_argument('url', help='URL to fetch and display')
    parser.add_argument(
        '--format', '-f',
        choices=['pretty', 'text', 'title', 'links', 'headings', 'images', 'all'],
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
    
    # Create the web page loader - like getting a smart web browser
    loader = WebPageLoader(timeout=args.timeout)
    
    # Fetch the page
    soup, response = loader.fetch_page(args.url)
    
    if not soup:
        print("Failed to load the page")
        sys.exit(1)
    
    print(f"Successfully loaded: {response.url}")
    print(f"Status code: {response.status_code}")
    print(f"Content type: {response.headers.get('content-type', 'Unknown')}")
    
    # Display based on requested format
    if args.format == 'pretty' or args.format == 'all':
        loader.display_pretty(soup)
    
    if args.format == 'text' or args.format == 'all':
        loader.display_text_only(soup)
    
    if args.format == 'title' or args.format == 'all':
        loader.display_title_and_meta(soup)
    
    if args.format == 'headings' or args.format == 'all':
        loader.display_headings(soup)
    
    if args.format == 'links' or args.format == 'all':
        loader.display_links(soup, response.url)
    
    if args.format == 'images' or args.format == 'all':
        loader.display_images(soup, response.url)

if __name__ == "__main__":
    main()

