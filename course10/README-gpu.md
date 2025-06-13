Looking at your code, the model and data automatically moved to GPU because of how PyTorch and Hugging Face `Trainer` work by default. Here's the flow:

## Automatic GPU Detection and Usage

Think of it like a smart parking system - if PyTorch finds an available GPU "parking spot," it automatically uses it without you having to specify where to park.

### What Happens Behind the Scenes

1. **PyTorch checks for CUDA availability** when you load the model:
```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
# PyTorch sees CUDA is available and notes it for later use
```

2. **Trainer automatically uses GPU** when it finds one:
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
)
# Trainer detects GPU availability and moves model + data there
```

## How to Control GPU Usage

### Option 1: Force CPU-only Training
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs from PyTorch

# Or in TrainingArguments:
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    no_cuda=True,  # Force CPU usage
)
```

### Option 2: Specify Which GPU
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

# Or manually move model:
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Option 3: Check What's Happening
```python
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
print(f"Model device: {next(model.parameters()).device}")
```

## Why Your Code Went to GPU

Your code has this line at the top:
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

This confirms CUDA is available. Then the `Trainer` class automatically:
1. Detects CUDA availability
2. Moves the model to GPU memory
3. Processes batches on GPU during training

It's like having a valet service - when you arrive with your car (model), they automatically park it in the premium spot (GPU) if one's available, without you having to ask.

