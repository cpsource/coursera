import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LaTeX text to split
latex_text = """\\documentclass{article}
\\begin{document}
\\maketitle
\\section{Introduction}
Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.
\\subsection{History of LLMs}
The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.
\\subsection{Applications of LLMs}
LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.
\\end{document}"""

def create_text_splitter_screenshot():
    """Create a screenshot showing the text splitting code and results"""
    
    # Text splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "\\section", "\\subsection", " ", ""]
    )
    
    # Create a document object and split it
    doc = Document(page_content=latex_text, metadata={"source": "latex_example"})
    chunks = text_splitter.split_documents([doc])
    
    # Create the figure with subplots
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle('LaTeX Text Splitter - Code and Results', fontsize=18, fontweight='bold', y=0.98)
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2], hspace=0.3, wspace=0.2)
    
    # 1. Original LaTeX Code (top left)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('📄 Original LaTeX Text', fontsize=14, fontweight='bold', pad=20)
    
    # Format the LaTeX text for display
    wrapped_latex = textwrap.fill(latex_text, width=120)
    ax1.text(0.02, 0.95, wrapped_latex, transform=ax1.transAxes, fontsize=9, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Splitter Code (middle)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('🔧 Text Splitter Configuration Code', fontsize=14, fontweight='bold', pad=20)
    
    splitter_code = """from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,                                    # Maximum characters per chunk
    chunk_overlap=50,                                  # Overlap between chunks
    length_function=len,                               # Function to measure length
    separators=["\\n\\n", "\\n", "\\\\section", "\\\\subsection", " ", ""]  # Split priorities
)

# Create document and split
doc = Document(page_content=latex_text, metadata={"source": "latex_example"})
chunks = text_splitter.split_documents([doc])

print(f"Original text length: {len(latex_text)} characters")
print(f"Number of chunks created: {len(chunks)}")"""
    
    ax2.text(0.02, 0.95, splitter_code, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # 3. Results - Split Chunks (bottom)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_title(f'📊 Splitting Results - {len(chunks)} Chunks Created', fontsize=14, fontweight='bold', pad=20)
    
    # Display each chunk
    y_pos = 0.95
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightcyan']
    
    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        chunk_text = chunk.page_content
        chunk_length = len(chunk_text)
        
        # Create header for each chunk
        header = f"Chunk {i+1} (Length: {chunk_length} chars)"
        ax3.text(0.02, y_pos, header, transform=ax3.transAxes, fontsize=11, 
                fontweight='bold', color='darkblue')
        y_pos -= 0.05
        
        # Wrap and display chunk content
        wrapped_chunk = textwrap.fill(chunk_text, width=100)
        chunk_height = len(wrapped_chunk.split('\n')) * 0.025
        
        # Add background rectangle
        rect = Rectangle((0.01, y_pos - chunk_height), 0.98, chunk_height,
                        facecolor=color, alpha=0.3, transform=ax3.transAxes)
        ax3.add_patch(rect)
        
        ax3.text(0.02, y_pos, wrapped_chunk, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        y_pos -= chunk_height + 0.05
        
        # Stop if we run out of space
        if y_pos < 0.1:
            if i < len(chunks) - 1:
                ax3.text(0.02, y_pos, f"... and {len(chunks) - i - 1} more chunks", 
                        transform=ax3.transAxes, fontsize=10, style='italic')
            break
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Add summary statistics
    stats_text = f"""📈 Summary Statistics:
• Original text: {len(latex_text)} characters
• Chunks created: {len(chunks)}
• Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars
• Chunk size range: {min(len(c.page_content) for c in chunks)} - {max(len(c.page_content) for c in chunks)} chars"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", alpha=0.8))
    
    # Save the screenshot
    plt.savefig('code_splitter.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ Screenshot saved as 'code_splitter.png'")
    print(f"📊 Split {len(latex_text)} characters into {len(chunks)} chunks")
    
    # Print chunk details to console
    print("\n🔍 Detailed Chunk Analysis:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.page_content)} chars")
        print(f"Preview: {chunk.page_content[:100]}...")
        print("-" * 50)
    
    # Show the plot
    plt.show()
    
    return chunks

def demonstrate_splitting_parameters():
    """Demonstrate different splitting parameters"""
    print("🧪 Demonstrating different splitter configurations:")
    
    configs = [
        {"chunk_size": 200, "chunk_overlap": 20, "name": "Small chunks"},
        {"chunk_size": 400, "chunk_overlap": 50, "name": "Medium chunks"},
        {"chunk_size": 600, "chunk_overlap": 100, "name": "Large chunks"}
    ]
    
    for config in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len
        )
        
        doc = Document(page_content=latex_text)
        chunks = splitter.split_documents([doc])
        
        print(f"\n{config['name']} (size={config['chunk_size']}, overlap={config['chunk_overlap']}):")
        print(f"  Created {len(chunks)} chunks")
        print(f"  Avg size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

if __name__ == "__main__":
    # Create the main screenshot
    chunks = create_text_splitter_screenshot()
    
    # Demonstrate different parameters
    demonstrate_splitting_parameters()
