import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
import os
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

def create_embedding_screenshot():
    """Create a screenshot showing Watsonx embedding code and results"""
    
    # Query to embed
    query = "How are you?"
    
    try:
        # Setup Watsonx embeddings (using your existing credentials)
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
        
        watsonx_embedding = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=os.environ.get('IBM_PROJECT_ID'),
            apikey=os.environ.get('IBM_API_KEY'),
            params=embed_params,
        )
        
        # Generate embedding
        print(f"🔍 Generating embedding for: '{query}'")
        embedding_vector = watsonx_embedding.embed_query(query)
        
        # Create the figure
        fig = plt.figure(figsize=(14, 16))
        fig.suptitle('Watsonx Embedding - Code and Results', fontsize=18, fontweight='bold', y=0.97)
        
        # Create grid layout
        gs = fig.add_gridspec(4, 1, height_ratios=[0.8, 1.2, 0.8, 1], hspace=0.4)
        
        # 1. Query Input (top)
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title('🎯 Input Query', fontsize=14, fontweight='bold', pad=20)
        
        query_text = f'query = "How are you?"'
        ax1.text(0.5, 0.5, query_text, transform=ax1.transAxes, fontsize=16, 
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.8))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # 2. Embedding Code (middle-top)
        ax2 = fig.add_subplot(gs[1])
        ax2.set_title('🔧 Watsonx Embedding Code', fontsize=14, fontweight='bold', pad=20)
        
        embedding_code = """import os
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# Configure embedding parameters
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

# Initialize Watsonx embedding model
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",          # Embedding model
    url="https://us-south.ml.cloud.ibm.com",          # Watsonx URL
    project_id=os.environ.get('IBM_PROJECT_ID'),      # Your project ID
    apikey=os.environ.get('IBM_API_KEY'),             # Your API key
    params=embed_params,                               # Parameters
)

# Generate embedding for the query
query = "How are you?"
embedding_vector = watsonx_embedding.embed_query(query)

print(f"Query: {query}")
print(f"Embedding dimension: {len(embedding_vector)}")
print(f"First 5 values: {embedding_vector[:5]}")"""
        
        ax2.text(0.02, 0.98, embedding_code, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # 3. Execution Results (middle-bottom)
        ax3 = fig.add_subplot(gs[2])
        ax3.set_title('📊 Execution Output', fontsize=14, fontweight='bold', pad=20)
        
        # Format the first 5 embedding values
        first_five = embedding_vector[:5]
        first_five_str = [f"{val:.6f}" for val in first_five]
        
        results_text = f"""Query: "How are you?"
Embedding dimension: {len(embedding_vector)}
First 5 embedding values: [{', '.join(first_five_str)}]"""
        
        ax3.text(0.02, 0.5, results_text, transform=ax3.transAxes, fontsize=12,
                fontfamily='monospace', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 4. Visual Representation of First 5 Values (bottom)
        ax4 = fig.add_subplot(gs[3])
        ax4.set_title('📈 Visual Representation of First 5 Embedding Values', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Create bar chart of first 5 values
        x_pos = np.arange(5)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        bars = ax4.bar(x_pos, first_five, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, first_five)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (max(first_five) - min(first_five)) * 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_xlabel('Embedding Dimension Index', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Embedding Value', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Dim {i}' for i in range(5)])
        ax4.grid(True, alpha=0.3)
        ax4.set_axisbelow(True)
        
        # Add statistics box
        stats_text = f"""📊 Embedding Statistics:
• Model: ibm/slate-125m-english-rtrvr
• Total dimensions: {len(embedding_vector)}
• Min value: {min(embedding_vector):.6f}
• Max value: {max(embedding_vector):.6f}
• Mean value: {np.mean(embedding_vector):.6f}
• Std deviation: {np.std(embedding_vector):.6f}"""
        
        fig.text(0.02, 0.15, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", alpha=0.8))
        
        # Add model info box
        model_info = f"""🤖 Model Information:
• Model ID: ibm/slate-125m-english-rtrvr
• Provider: IBM Watsonx
• Type: Retrieval-focused embedding
• Input: "{query}"
• Output: {len(embedding_vector)}-dimensional vector"""
        
        fig.text(0.52, 0.15, model_info, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.8))
        
        # Save the screenshot
        plt.savefig('embedding.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print("✅ Screenshot saved as 'embedding.png'")
        print(f"📊 Generated embedding with {len(embedding_vector)} dimensions")
        print(f"🎯 First 5 values: {first_five}")
        
        # Show the plot
        plt.show()
        
        return embedding_vector
        
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        # Create error screenshot
        create_error_screenshot(str(e))
        return None

def create_error_screenshot(error_msg):
    """Create screenshot showing the code even if embedding fails"""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Watsonx Embedding Code (Error Occurred)', fontsize=18, fontweight='bold')
    
    # Show the code that would be used
    embedding_code = """import os
from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# Configure embedding parameters
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

# Initialize Watsonx embedding model
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.environ.get('IBM_PROJECT_ID'),
    apikey=os.environ.get('IBM_API_KEY'),
    params=embed_params,
)

# Generate embedding for the query
query = "How are you?"
embedding_vector = watsonx_embedding.embed_query(query)

# Display results
print(f"Query: {query}")
print(f"Embedding dimension: {len(embedding_vector)}")
print(f"First 5 values: {embedding_vector[:5]}")"""
    
    ax = fig.add_subplot(111)
    ax.text(0.02, 0.98, embedding_code, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    # Add error message
    ax.text(0.02, 0.25, f"❌ Error: {error_msg}", transform=ax.transAxes, 
            fontsize=12, color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.9))
    
    ax.text(0.02, 0.15, "💡 Note: Code is shown above. Check credentials and try again.", 
            transform=ax.transAxes, fontsize=11, style='italic')
    
    ax.axis('off')
    plt.savefig('embedding.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("⚠️ Error screenshot saved as 'embedding.png'")

def test_embedding_simple():
    """Simple test function to verify embedding works"""
    query = "How are you?"
    
    try:
        # Your existing embedding setup
        from dotenv import load_dotenv
        load_dotenv()  # Load credentials
        
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
        
        watsonx_embedding = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=os.environ.get('IBM_PROJECT_ID'),
            apikey=os.environ.get('IBM_API_KEY'),
            params=embed_params,
        )
        
        embedding = watsonx_embedding.embed_query(query)
        print(f"✅ Test successful! Embedding has {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Make sure credentials are loaded
    print("🔍 Testing embedding functionality...")
    
    # Test first
    if test_embedding_simple():
        print("✅ Test passed, creating screenshot...")
        embedding_vector = create_embedding_screenshot()
    else:
        print("⚠️ Test failed, but creating code screenshot anyway...")
        create_error_screenshot("Credential or connection issue")
