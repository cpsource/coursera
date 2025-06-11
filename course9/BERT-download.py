# BERT Model Cache Locations and Management

import os
import torch
from transformers import BertTokenizer, BertForPreTraining
from pathlib import Path
import shutil

def show_cache_locations():
    """Show where Transformers models are cached"""
    print("🗂️  BERT Model Cache Locations")
    print("=" * 50)
    
    # Default cache directory
    from transformers import TRANSFORMERS_CACHE
    print(f"📁 Default Transformers cache: {TRANSFORMERS_CACHE}")
    
    # Alternative ways to find cache
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', 
                              os.environ.get('HF_HOME', 
                                           os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')))
    print(f"📁 Computed cache directory: {cache_dir}")
    
    # Platform-specific defaults
    if os.name == 'nt':  # Windows
        default_cache = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        print(f"🪟 Windows default: {default_cache}")
    else:  # Linux/Mac
        default_cache = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        print(f"🐧 Linux/Mac default: {default_cache}")
    
    print(f"\n📍 Your home directory: {os.path.expanduser('~')}")

def explore_cache_contents():
    """Explore what's actually in the cache"""
    print("\n🔍 Exploring Cache Contents")
    print("=" * 30)
    
    # Get the cache directory
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', 
                              os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    
    transformers_cache = os.path.join(cache_dir, 'transformers')
    
    if os.path.exists(transformers_cache):
        print(f"✅ Cache directory exists: {transformers_cache}")
        
        # List contents
        try:
            contents = os.listdir(transformers_cache)
            print(f"📦 Number of cached files: {len(contents)}")
            
            # Show first few files
            for i, item in enumerate(contents[:5]):
                item_path = os.path.join(transformers_cache, item)
                size = os.path.getsize(item_path) / (1024*1024)  # MB
                print(f"   {i+1}. {item[:50]}... ({size:.1f} MB)")
            
            if len(contents) > 5:
                print(f"   ... and {len(contents) - 5} more files")
                
        except PermissionError:
            print("❌ Permission denied to read cache directory")
        except Exception as e:
            print(f"❌ Error reading cache: {e}")
    else:
        print(f"❌ Cache directory doesn't exist yet: {transformers_cache}")

def download_and_show_bert_location():
    """Download BERT and show exactly where it's stored"""
    print("\n📥 Downloading BERT and Showing Location")
    print("=" * 40)
    
    # This will download BERT if not already cached
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Loading BERT model...")
    model = BertForPreTraining.from_pretrained('bert-base-uncased')
    
    # Show model info
    print(f"✅ BERT model loaded successfully")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Try to find the specific BERT files
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', 
                              os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    
    transformers_cache = os.path.join(cache_dir, 'transformers')
    
    if os.path.exists(transformers_cache):
        # Look for BERT-related files
        bert_files = []
        for file in os.listdir(transformers_cache):
            if 'bert' in file.lower() or any(x in file for x in ['pytorch_model', 'config.json', 'vocab.txt']):
                bert_files.append(file)
        
        print(f"\n🔍 Found {len(bert_files)} BERT-related files:")
        for file in bert_files[:10]:  # Show first 10
            file_path = os.path.join(transformers_cache, file)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   📄 {file[:60]}... ({size:.1f} MB)")

def get_cache_size():
    """Calculate total cache size"""
    print("\n📏 Cache Size Information")
    print("=" * 25)
    
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', 
                              os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    
    if os.path.exists(cache_dir):
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except (OSError, FileNotFoundError):
                    pass
        
        total_size_gb = total_size / (1024**3)
        print(f"📦 Total cache size: {total_size_gb:.2f} GB")
        print(f"📄 Total files: {file_count}")
        
        # Breakdown by subdirectory
        for subdir in ['transformers', 'datasets', 'hub']:
            subdir_path = os.path.join(cache_dir, subdir)
            if os.path.exists(subdir_path):
                subdir_size = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, dirs, files in os.walk(subdir_path)
                    for file in files
                )
                subdir_size_gb = subdir_size / (1024**3)
                print(f"   📁 {subdir}: {subdir_size_gb:.2f} GB")
    else:
        print("❌ Cache directory not found")

def clear_cache_options():
    """Show options for clearing cache"""
    print("\n🧹 Cache Management Options")
    print("=" * 28)
    
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', 
                              os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    
    print(f"🗑️  To clear all cached models:")
    print(f"   rm -rf {cache_dir}")
    print(f"   # OR on Windows: rmdir /s {cache_dir}")
    
    print(f"\n📂 To clear only transformers cache:")
    transformers_cache = os.path.join(cache_dir, 'transformers')
    print(f"   rm -rf {transformers_cache}")
    
    print(f"\n🔧 To set custom cache location:")
    print(f"   export TRANSFORMERS_CACHE=/your/custom/path")
    print(f"   # OR on Windows: set TRANSFORMERS_CACHE=C:\\your\\custom\\path")
    
    print(f"\n💡 Programmatic cache clearing:")
    print(f"   import shutil")
    print(f"   shutil.rmtree('{cache_dir}')")

def demonstrate_custom_cache():
    """Show how to use a custom cache directory"""
    print("\n⚙️  Custom Cache Directory Example")
    print("=" * 35)
    
    # Create a custom cache directory
    custom_cache = "./my_bert_models"
    os.makedirs(custom_cache, exist_ok=True)
    
    print(f"📁 Created custom cache: {os.path.abspath(custom_cache)}")
    
    # Set environment variable
    os.environ['TRANSFORMERS_CACHE'] = custom_cache
    
    print("📥 Downloading BERT to custom location...")
    
    # This will now use the custom cache
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Check what was downloaded
    if os.path.exists(custom_cache):
        files = os.listdir(custom_cache)
        print(f"✅ Files in custom cache: {len(files)}")
        for file in files[:3]:
            print(f"   📄 {file}")
    
    # Reset to default
    if 'TRANSFORMERS_CACHE' in os.environ:
        del os.environ['TRANSFORMERS_CACHE']
    
    print("🔄 Reset to default cache location")

def show_model_files_breakdown():
    """Show what each file in BERT download contains"""
    print("\n📋 BERT Model Files Breakdown")
    print("=" * 32)
    
    file_descriptions = {
        'config.json': 'Model configuration (architecture, vocab size, etc.)',
        'pytorch_model.bin': 'Main model weights (largest file ~440MB)',
        'tokenizer_config.json': 'Tokenizer configuration',
        'tokenizer.json': 'Fast tokenizer data',
        'vocab.txt': 'Vocabulary file (~230KB)',
        'special_tokens_map.json': 'Special token mappings ([CLS], [SEP], etc.)'
    }
    
    print("📂 BERT-base-uncased typically contains:")
    for filename, description in file_descriptions.items():
        print(f"   📄 {filename:<25} - {description}")
    
    print(f"\n💾 Total size: ~440MB")
    print(f"🔗 Files are linked by hash to prevent duplicates")

if __name__ == "__main__":
    print("🤗 BERT Model Storage Location Guide")
    print("=" * 50)
    
    # Show where models are cached
    show_cache_locations()
    
    # Explore existing cache
    explore_cache_contents()
    
    # Download BERT and show location
    download_and_show_bert_location()
    
    # Show cache size
    get_cache_size()
    
    # Show file breakdown
    show_model_files_breakdown()
    
    # Show cache management options
    clear_cache_options()
    
    # Demonstrate custom cache
    print("\n" + "="*60)
    demonstrate_custom_cache()
    
    print(f"\n✅ Complete! Check the locations above to find your BERT models.")

