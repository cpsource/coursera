# Debug script to check library versions and fix hnswlib issue
import subprocess
import os
import sys

def check_library_versions():
    """Check what GLIBCXX versions are available"""
    print("=== LIBRARY VERSION CHECK ===")
    
    # Check system libstdc++
    try:
        result = subprocess.run(['strings', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            glibcxx_versions = [line for line in result.stdout.split('\n') if 'GLIBCXX' in line]
            print("System libstdc++ GLIBCXX versions:")
            for version in sorted(set(glibcxx_versions))[-10:]:  # Last 10
                print(f"  {version}")
        else:
            print("Could not read system libstdc++")
    except Exception as e:
        print(f"Error checking system libstdc++: {e}")
    
    # Check conda libstdc++
    conda_lib = "/home/ubuntu/miniconda3/envs/nlp_course1/lib/libstdc++.so.6"
    if os.path.exists(conda_lib):
        try:
            result = subprocess.run(['strings', conda_lib], capture_output=True, text=True)
            if result.returncode == 0:
                glibcxx_versions = [line for line in result.stdout.split('\n') if 'GLIBCXX' in line]
                print(f"\nConda libstdc++ GLIBCXX versions:")
                for version in sorted(set(glibcxx_versions))[-10:]:  # Last 10
                    print(f"  {version}")
            else:
                print("Could not read conda libstdc++")
        except Exception as e:
            print(f"Error checking conda libstdc++: {e}")
    else:
        print(f"Conda libstdc++ not found at {conda_lib}")
    
    # Check LD_LIBRARY_PATH
    print(f"\nLD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")

def fix_hnswlib_compatibility():
    """Try different approaches to fix hnswlib"""
    print("\n=== FIXING HNSWLIB ===")
    
    # Option 1: Update conda libstdc++
    print("Option 1: Updating conda libstdc++...")
    try:
        result = subprocess.run([
            'conda', 'update', '-c', 'conda-forge', 'libstdcxx-ng', 'libgcc-ng', '-y'
        ], capture_output=True, text=True, cwd='/home/ubuntu')
        
        if result.returncode == 0:
            print("✅ Successfully updated conda C++ libraries")
            return True
        else:
            print(f"❌ Failed to update: {result.stderr}")
    except Exception as e:
        print(f"❌ Error updating conda libraries: {e}")
    
    # Option 2: Reinstall hnswlib with conda
    print("\nOption 2: Reinstalling hnswlib with conda...")
    try:
        # Remove pip version
        subprocess.run(['pip', 'uninstall', 'hnswlib', '-y'], 
                      capture_output=True, text=True)
        
        # Install conda version
        result = subprocess.run([
            'conda', 'install', '-c', 'conda-forge', 'hnswlib', '-y'
        ], capture_output=True, text=True, cwd='/home/ubuntu')
        
        if result.returncode == 0:
            print("✅ Successfully installed hnswlib via conda")
            return True
        else:
            print(f"❌ Failed to install via conda: {result.stderr}")
    except Exception as e:
        print(f"❌ Error installing hnswlib via conda: {e}")
    
    return False

def create_faiss_alternative():
    """Create alternative vector_database function using FAISS"""
    faiss_code = '''
def vector_database(chunks):
    """Create vector database using FAISS instead of Chroma"""
    try:
        from langchain_community.vectorstores import FAISS
        print("🔄 Using FAISS instead of Chroma to avoid hnswlib issues")
        
        embedding_model = watsonx_embedding()
        vectordb = FAISS.from_documents(chunks, embedding_model)
        print("✅ Created FAISS vector database successfully")
        return vectordb
    except Exception as e:
        print(f"❌ Error creating FAISS vector database: {e}")
        raise
'''
    
    print("\n=== FAISS ALTERNATIVE ===")
    print("Replace your vector_database function with this:")
    print(faiss_code)
    
    # Check if FAISS is available
    try:
        import faiss
        print("✅ FAISS is already installed")
    except ImportError:
        print("⚠️  FAISS not installed. Installing...")
        try:
            subprocess.run(['pip', 'install', 'faiss-cpu'], check=True)
            print("✅ FAISS installed successfully")
        except Exception as e:
            print(f"❌ Failed to install FAISS: {e}")

if __name__ == "__main__":
    print("🔍 Diagnosing hnswlib/libstdc++ compatibility issue...")
    
    # Check current state
    check_library_versions()
    
    # Try to fix
    if not fix_hnswlib_compatibility():
        print("\n⚠️  Could not fix hnswlib directly. Using FAISS alternative...")
        create_faiss_alternative()
    
    print("\n🔄 After making changes, restart your Python session and try again.")
