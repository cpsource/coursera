from datasets import load_dataset

# Load the specific split from the dataset
dataset = load_dataset("BarraHome/ultrafeedback_binarized", split="train_prefs")

# Fix 1: Access info from dataset_info, not info directly
print("Builder name:", dataset.info.builder_name)

# Access and inspect the first example
first_example = dataset[0]
print("Keys in first example:", first_example.keys())

# Work with the already loaded train_prefs split
# Get builder info from the split we already have
builder_name = dataset.info.builder_name

# Handle builder_config - for most datasets, we can construct the URL directly
# without needing the builder_config
if hasattr(dataset, 'builder_config') and dataset.builder_config:
    builder_config = dataset.builder_config.name
else:
    # Most datasets don't need a specific config in the URL
    builder_config = None

# Fix 5: Construct URL using the actual dataset path from HuggingFace
# The correct format is: https://huggingface.co/datasets/{username}/{dataset_name}
dataset_path = "BarraHome/ultrafeedback_binarized"
url = f"https://huggingface.co/datasets/{dataset_path}"
print("Dataset URL:", url)

