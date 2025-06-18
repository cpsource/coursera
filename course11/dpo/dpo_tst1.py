from datasets import load_dataset

# Load the specific split from the dataset
dataset = load_dataset("BarraHome/ultrafeedback_binarized", split="train_prefs")

# Fix 1: Access info from dataset_info, not info directly
print("Builder name:", dataset.info.builder_name)

# Access and inspect the first example
first_example = dataset[0]
print("Keys in first example:", first_example.keys())

# Fix 2: Load the full dataset to access builder_config properly
full_dataset = load_dataset("BarraHome/ultrafeedback_binarized")

# Fix 3: Access builder_name from the full dataset's info
builder_name = full_dataset.info.builder_name

# Fix 4: Handle builder_config more robustly
# Some datasets don't have a specific config name, so we use the dataset name
if hasattr(full_dataset, 'builder_config') and full_dataset.builder_config:
    builder_config = full_dataset.builder_config.name
else:
    # Fallback: use the dataset name from the path
    builder_config = "ultrafeedback_binarized"

# Fix 5: Construct URL using the actual dataset path from HuggingFace
# The correct format is: https://huggingface.co/datasets/{username}/{dataset_name}
dataset_path = "BarraHome/ultrafeedback_binarized"
url = f"https://huggingface.co/datasets/{dataset_path}"
print("Dataset URL:", url)

