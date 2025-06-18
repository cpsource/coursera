from datasets import load_dataset

# Load the full dataset (all splits)
dataset = load_dataset("BarraHome/ultrafeedback_binarized")

# Access info from the first available split (since dataset is now a DatasetDict)
first_split_name = list(dataset.keys())[0]
first_split = dataset[first_split_name]

# Access builder info and inspect first example
print("Builder name:", first_split.info.builder_name)

# Access and inspect the first example from the first split
first_example = first_split[0]
print("Keys in first example:", first_example.keys())

# Print out the complete first record
print("\n=== First Record ===")
for key, value in first_example.items():
    print(f"{key}:")
    if isinstance(value, str) and len(value) > 200:
        # Truncate very long strings for readability
        print(f"  {value[:200]}...")
    else:
        print(f"  {value}")
    print()  # Add blank line between fields

# Work with the first split for metadata
# Get builder info from any split (they all have the same metadata)
builder_name = first_split.info.builder_name

# Handle builder_config - for most datasets, we can construct the URL directly
# without needing the builder_config
if hasattr(first_split, 'builder_config') and first_split.builder_config:
    builder_config = first_split.builder_config.name
else:
    # Most datasets don't need a specific config in the URL
    builder_config = None

# Fix 5: Construct URL using the actual dataset path from HuggingFace
# The correct format is: https://huggingface.co/datasets/{username}/{dataset_name}
dataset_path = "BarraHome/ultrafeedback_binarized"
url = f"https://huggingface.co/datasets/{dataset_path}"
print("Dataset URL:", url)

