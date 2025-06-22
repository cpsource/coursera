import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import textwrap
import os
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# The new-Policies.txt content
policies_content = """1. Code of Conduct

Our Code of Conduct establishes the core values and ethical standards that all members of our organization must adhere to. We are committed to fostering a workplace characterized by integrity, respect, and accountability.

Integrity: We commit to the highest ethical standards by being honest and transparent in all our dealings, whether with colleagues, clients, or the community. We protect sensitive information and avoid conflicts of interest.

Respect: We value diversity and every individual's contribution. Discrimination, harassment, or any form of disrespect is not tolerated. We promote an inclusive environment where differences are respected, and everyone is treated with dignity.

Accountability: We are responsible for our actions and decisions, complying with all relevant laws and regulations. We aim for continuous improvement and report any breaches of this code, supporting investigations into such matters.

Safety: We prioritize the safety of our employees, clients, and the community. We encourage a culture of safety, including reporting any unsafe practices or conditions.

Environmental Responsibility: We strive to reduce our environmental impact and promote sustainable practices.

This Code of Conduct is the cornerstone of our organizational culture. We expect every employee to uphold these principles and act as role models, ensuring our reputation for ethical conduct, integrity, and social responsibility.

2. Recruitment Policy

Our Recruitment Policy is dedicated to attracting, selecting, and integrating the most qualified and diverse candidates into our organization. The success of our company depends on the talent, skills, and commitment of our employees.

Equal Opportunity: We are an equal opportunity employer and do not discriminate based on race, color, religion, sex, sexual orientation, gender identity, national origin, age, disability, or any other protected status. We actively support diversity and inclusion.

Transparency: We maintain a transparent recruitment process. Job vacancies are advertised both internally and externally when appropriate. Job descriptions and requirements are clear and accurately reflect the role.

Selection Criteria: We base our selection on qualifications, experience, and skills relevant to the role. Our interviews and assessments are objective, and decisions are made impartially.

Data Privacy: We are dedicated to protecting candidates' personal information and comply with all applicable data protection laws.

Feedback: Candidates receive timely and constructive feedback on their applications and interview performance.

Onboarding: New hires receive thorough onboarding to help them integrate effectively, including an overview of our culture, policies, and expectations.

Employee Referrals: We welcome employee referrals as they help build a strong and engaged team.

This policy lays the foundation for a diverse, inclusive, and talented workforce. It ensures that we hire candidates who align with our values and contribute to our success. We regularly review and update this policy to incorporate best practices in recruitment.

3. Internet and Email Policy

Our Internet and Email Policy ensures the responsible and secure use of these tools within our organization, recognizing their importance in daily operations and the need for compliance with security, productivity, and legal standards.

Acceptable Use: Company-provided internet and email are primarily for job-related tasks. Limited personal use is permitted during non-work hours as long as it does not interfere with work duties.

Security: Protect your login credentials and avoid sharing passwords. Be cautious with email attachments and links from unknown sources, and promptly report any unusual online activity or potential security threats.

Confidentiality: Use email for confidential information, trade secrets, and sensitive customer data only with encryption. Be careful when discussing company matters on public platforms or social media.

Harassment and Inappropriate Content: Internet and email must not be used for harassment, discrimination, or the distribution of offensive content. Always communicate respectfully and sensitively online.

Compliance: Adhere to all relevant laws and regulations concerning internet and email use, including copyright and data protection laws.

Monitoring: The company reserves the right to monitor internet and email usage for security and compliance purposes.

Consequences: Violations of this policy may lead to disciplinary action, including potential termination.

This policy promotes the safe and responsible use of digital communication tools in line with our values and legal obligations. Employees must understand and comply with this policy. Regular reviews will ensure it remains relevant with changing technology and security standards.

4. Mobile Phone Policy

Our Mobile Phone Policy defines standards for responsible use of mobile devices within the organization to ensure alignment with company values and legal requirements.

Acceptable Use: Mobile devices are primarily for work-related tasks. Limited personal use is allowed if it does not disrupt work responsibilities.

Security: Secure your mobile device and credentials. Be cautious with app downloads and links from unknown sources, and report any security issues promptly.

Confidentiality: Avoid sharing sensitive company information via unsecured messaging apps or emails. Exercise caution when discussing company matters in public.

Cost Management: Personal use of mobile phones should be separate from company accounts, and any personal charges on company-issued phones must be reimbursed.

Compliance: Comply with all relevant laws and regulations concerning mobile phone usage, including data protection and privacy laws.

Lost or Stolen Devices: Immediately report any lost or stolen mobile devices to the IT department or your supervisor.

Consequences: Non-compliance with this policy may result in disciplinary actions, including potential loss of mobile phone privileges.

This policy encourages the responsible use of mobile devices in line with legal and ethical standards. Employees are expected to understand and follow these guidelines. The policy is regularly reviewed to stay current with evolving technology and security best practices."""

def create_vectordb_screenshot():
    """Create a screenshot showing Chroma vector database creation and similarity search"""
    
    query = "Smoking policy"
    
    try:
        # Setup Watsonx embeddings
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
        
        # Create document and split into chunks
        document = Document(page_content=policies_content, metadata={"source": "new-Policies.txt"})
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents([document])
        
        # Create Chroma vector database
        print(f"🔍 Creating Chroma vector database with {len(chunks)} chunks...")
        vectordb = Chroma.from_documents(chunks, watsonx_embedding)
        
        # Perform similarity search
        print(f"🔍 Searching for: '{query}'")
        search_results = vectordb.similarity_search(query, k=5)
        
        # Create the figure
        fig = plt.figure(figsize=(16, 20))
        fig.suptitle('Chroma Vector Database - Creation and Similarity Search', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Create grid layout
        gs = fig.add_gridspec(5, 1, height_ratios=[0.6, 1.4, 0.6, 1.8, 0.8], hspace=0.3)
        
        # 1. Document Input (top)
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title('📄 Input Document: new-Policies.txt', fontsize=14, fontweight='bold', pad=20)
        
        doc_preview = policies_content[:200] + "..."
        wrapped_preview = textwrap.fill(doc_preview, width=120)
        ax1.text(0.02, 0.8, wrapped_preview, transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        
        ax1.text(0.02, 0.2, f"📊 Total document length: {len(policies_content)} characters", 
                transform=ax1.transAxes, fontsize=11, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # 2. Vector Database Creation Code (middle-top)
        ax2 = fig.add_subplot(gs[1])
        ax2.set_title('🔧 Chroma Vector Database Creation Code', fontsize=14, fontweight='bold', pad=20)
        
        vectordb_code = """import os
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# 1. Setup Watsonx embeddings
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

# 2. Load and split document
with open('new-Policies.txt', 'r') as f:
    content = f.read()

document = Document(page_content=content, metadata={"source": "new-Policies.txt"})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

chunks = text_splitter.split_documents([document])

# 3. Create Chroma vector database
vectordb = Chroma.from_documents(chunks, watsonx_embedding)

print(f"✅ Created vector database with {len(chunks)} chunks")"""
        
        ax2.text(0.02, 0.98, vectordb_code, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # 3. Similarity Search Code (middle)
        ax3 = fig.add_subplot(gs[2])
        ax3.set_title('🔍 Similarity Search Code', fontsize=14, fontweight='bold', pad=20)
        
        search_code = """# 4. Perform similarity search
query = "Smoking policy"
search_results = vectordb.similarity_search(query, k=5)

print(f"Query: {query}")
print(f"Found {len(search_results)} results:")
for i, result in enumerate(search_results):
    print(f"Result {i+1}: {result.page_content[:100]}...")"""
        
        ax3.text(0.02, 0.5, search_code, transform=ax3.transAxes, fontsize=11,
                fontfamily='monospace', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # 4. Search Results (bottom-main)
        ax4 = fig.add_subplot(gs[3])
        ax4.set_title(f'📊 Similarity Search Results for "{query}" (Top 5)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Display search results
        y_pos = 0.95
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        for i, result in enumerate(search_results[:5]):
            color = colors[i % len(colors)]
            content = result.page_content
            metadata = result.metadata
            
            # Result header
            header = f"Result {i+1} (Source: {metadata.get('source', 'unknown')})"
            ax4.text(0.02, y_pos, header, transform=ax4.transAxes, fontsize=12, 
                    fontweight='bold', color='darkblue')
            y_pos -= 0.04
            
            # Wrap content
            wrapped_content = textwrap.fill(content, width=90)
            content_lines = wrapped_content.split('\n')
            content_height = len(content_lines) * 0.025
            
            # Add background rectangle
            rect = Rectangle((0.01, y_pos - content_height), 0.98, content_height,
                           facecolor=color, alpha=0.3, transform=ax4.transAxes)
            ax4.add_patch(rect)
            
            # Display content
            ax4.text(0.02, y_pos, wrapped_content, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace')
            
            y_pos -= content_height + 0.06
            
            # Stop if we run out of space
            if y_pos < 0.1:
                break
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # 5. Execution Summary (bottom)
        ax5 = fig.add_subplot(gs[4])
        ax5.set_title('📈 Execution Summary', fontsize=14, fontweight='bold', pad=20)
        
        summary_text = f"""✅ Vector Database Creation Summary:
• Document: new-Policies.txt ({len(policies_content)} characters)
• Chunks created: {len(chunks)}
• Embedding model: ibm/slate-125m-english-rtrvr
• Vector database: Chroma
• Search query: "{query}"
• Results found: {len(search_results)}"""
        
        ax5.text(0.02, 0.7, summary_text, transform=ax5.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", alpha=0.8))
        
        # Analysis note
        analysis_text = f"""🔍 Search Analysis:
• Query: "{query}" 
• No explicit smoking policy found in document
• Results show most relevant policy sections
• Demonstrates semantic similarity search capabilities"""
        
        ax5.text(0.52, 0.7, analysis_text, transform=ax5.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.8))
        
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        
        # Save screenshot
        plt.savefig('vectordb.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print("✅ Screenshot saved as 'vectordb.png'")
        print(f"📊 Created vector database with {len(chunks)} chunks")
        print(f"🔍 Search for '{query}' returned {len(search_results)} results")
        
        # Print detailed results to console
        print(f"\n🔍 Detailed Search Results for '{query}':")
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Content: {result.page_content[:150]}...")
            print(f"Metadata: {result.metadata}")
            print("-" * 70)
        
        plt.show()
        
        return vectordb, search_results
        
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        create_error_screenshot(str(e))
        return None, None

def create_error_screenshot(error_msg):
    """Create screenshot showing the code even if there's an error"""
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Chroma Vector Database Code (Error Occurred)', fontsize=18, fontweight='bold')
    
    # Show the code that would be used
    code = """# Chroma Vector Database Creation and Search
import os
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames

# Setup embeddings and create vector database
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

# Load document and create vector database
document = Document(page_content=content, metadata={"source": "new-Policies.txt"})
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents([document])
vectordb = Chroma.from_documents(chunks, watsonx_embedding)

# Perform similarity search
query = "Smoking policy"
search_results = vectordb.similarity_search(query, k=5)"""
    
    ax = fig.add_subplot(111)
    ax.text(0.02, 0.95, code, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    # Add error message
    ax.text(0.02, 0.25, f"❌ Error: {error_msg}", transform=ax.transAxes, 
            fontsize=12, color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose", alpha=0.9))
    
    ax.text(0.02, 0.15, "💡 Note: Code shown above. Check credentials and connectivity.", 
            transform=ax.transAxes, fontsize=11, style='italic')
    
    ax.axis('off')
    plt.savefig('vectordb.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("⚠️ Error screenshot saved as 'vectordb.png'")

def save_policies_file():
    """Save the policies content to new-Policies.txt file"""
    with open('new-Policies.txt', 'w') as f:
        f.write(policies_content)
    print("📄 Saved new-Policies.txt file")

if __name__ == "__main__":
    # Save the policies file first
    save_policies_file()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("🔍 Creating Chroma vector database screenshot...")
        vectordb, results = create_vectordb_screenshot()
    except Exception as e:
        print(f"Error loading environment: {e}")
        create_error_screenshot("Environment setup issue")
