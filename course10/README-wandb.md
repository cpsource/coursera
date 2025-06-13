Looking at your code, I don't see any explicit Weights & Biases (wandb) integration, but the Hugging Face `Trainer` class automatically logs to wandb if it's installed in your environment. Here's what's happening:

## Why Trainer Sends Data to wandb

Think of the Hugging Face `Trainer` like a smart assistant that automatically documents your work. If it finds wandb installed (like finding a camera in the room), it assumes you want photos of your progress and starts taking them automatically.

The `Trainer` class has built-in integrations that activate when it detects certain logging libraries:
- **wandb** (Weights & Biases)
- **tensorboard** 
- **mlflow**
- **comet_ml**

## How to Prevent wandb Logging

Here are three approaches, like different ways to tell that assistant to stop taking photos:

### Option 1: Disable wandb in TrainingArguments
```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    report_to=[],  # Empty list = no logging to external services
    # OR
    report_to="none",  # Explicitly disable all reporting
)
```

### Option 2: Set Environment Variable
```python
import os
os.environ["WANDB_DISABLED"] = "true"

# Then your existing TrainingArguments work as-is
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
)
```

### Option 3: Disable wandb Globally
```python
import wandb
wandb.init(mode="disabled")
```

## Example: Your Code with wandb Disabled

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    report_to=[],  # This line prevents wandb logging
    logging_dir=None,  # Optional: also disable local logging
)
```

The `report_to=[]` parameter is like telling your assistant "don't document this anywhere" - it keeps all the training local to your machine without sending metrics to external services.

