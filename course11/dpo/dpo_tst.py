from datasets import load_dataset

# Load the specific split from the dataset
dataset = load_dataset("BarraHome/ultrafeedback_binarized", split="train_prefs")
print(dataset.info.builder_name)

# Access and inspect the first example
first_example = dataset[0]
print(first_example.keys())

from datasets import load_dataset

# Load the full dataset (all splits)
#dataset = load_dataset("BarraHome/ultrafeedback_binarized")

# Extract dataset ID from the builder config to construct the URL
builder_name = dataset.builder_name
builder_config = dataset.builder_config.name  # Usually just the dataset name

# Construct the Hugging Face URL
url = f"https://huggingface.co/datasets/{builder_name}/{builder_config}"
print("Dataset URL:", url)

