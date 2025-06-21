from langchain_text_splitters import HTMLHeaderTextSplitter

# Sample HTML content
html_content = """
<html>
<body>
    <h1>Introduction to Web Scraping</h1>
    <p>Web scraping is the process of extracting data from websites. It's like having a robot read web pages for you.</p>
    <p>There are many tools available for web scraping.</p>
    
    <h2>Popular Tools</h2>
    <p>Some popular tools include Beautiful Soup, Selenium, and Scrapy.</p>
    
    <h3>Beautiful Soup</h3>
    <p>Beautiful Soup is great for parsing HTML and XML documents.</p>
    <p>It creates a parse tree from page source code.</p>
    
    <h3>Selenium</h3>
    <p>Selenium can handle JavaScript-heavy websites.</p>
    
    <h2>Best Practices</h2>
    <p>Always respect robots.txt files.</p>
    <p>Don't overload servers with too many requests.</p>
    
    <h3>Rate Limiting</h3>
    <p>Add delays between requests to be respectful.</p>
</body>
</html>
"""

def demo_html_splitter():
    """Demonstrate how HTMLHeaderTextSplitter works"""
    
    # Define the headers to split on
    headers_to_split_on = [
        ("h1", "Header 1"),  # h1 tags will be labeled as "Header 1"
        ("h2", "Header 2"),  # h2 tags will be labeled as "Header 2"  
        ("h3", "Header 3"),  # h3 tags will be labeled as "Header 3"
    ]
    
    # Create the splitter
    html_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    # Split the HTML
    documents = html_splitter.split_text(html_content)
    
    print("HTMLHeaderTextSplitter Results:")
    print("=" * 50)
    
    for i, doc in enumerate(documents, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 30)

def explain_labels():
    """Explain what the labels mean"""
    print("\nLabel Explanation:")
    print("=" * 30)
    print("The second part of each tuple is a LABEL, not literal text:")
    print()
    print("('h1', 'Header 1') means:")
    print("  - Look for <h1> tags in the HTML")
    print("  - When found, add the h1's text to metadata with key 'Header 1'")
    print("  - The actual h1 text becomes the VALUE")
    print()
    print("So if HTML has: <h1>My Title</h1>")
    print("Metadata becomes: {'Header 1': 'My Title'}")
    print()
    print("You could use any labels you want:")
    print("  ('h1', 'Main Title')")
    print("  ('h2', 'Section')")  
    print("  ('h3', 'Subsection')")

def custom_labels_example():
    """Show example with custom labels"""
    print("\n" + "=" * 50)
    print("CUSTOM LABELS EXAMPLE")
    print("=" * 50)
    
    # Custom labels that make more sense
    custom_headers = [
        ("h1", "Chapter"),
        ("h2", "Section"), 
        ("h3", "Topic"),
    ]
    
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=custom_headers)
    docs = splitter.split_text(html_content)
    
    print("With custom labels:")
    for i, doc in enumerate(docs[:2], 1):  # Show first 2
        print(f"\nDocument {i}:")
        print(f"Metadata: {doc.metadata}")
        # Now you see "Chapter", "Section", "Topic" instead of "Header 1", etc.

if __name__ == "__main__":
    demo_html_splitter()
    explain_labels() 
    custom_labels_example()

