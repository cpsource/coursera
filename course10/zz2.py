from transformers import pipeline
from transformers.utils import TRANSFORMERS_CACHE
import os

print("Transformers cache location:", TRANSFORMERS_CACHE)
print("Cache exists:", os.path.exists(TRANSFORMERS_CACHE))

# Or you can check the default cache directory
from huggingface_hub import HfFolder
print("HF Hub cache:", HfFolder.get_cache_dir())

