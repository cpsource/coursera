# Check if model is already cached
from transformers import pipeline
print("Model cache location:", pipeline._get_cache_dir())  # Check what's there
