import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class LoRALinear(nn.Module):
    """
    LoRA adaptation for Linear layers
    Like adding a lightweight "skill patch" to an existing layer
    Instead of modifying W directly, we add A*B where A and B are low-rank matrices
    """
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Get dimensions from original layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA matrices: W + (alpha/rank) * A @ B
        # A: (out_features, rank), B: (rank, in_features)
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original computation + LoRA adaptation
        original_output = self.original_layer(x)
        lora_output = (self.alpha / self.rank) * (x @ self.lora_B.T @ self.lora_A.T)
        return original_output + lora_output

# Your model setup (assuming NeuralNetwork class is defined)
exercise_model = NeuralNetwork()
exercise_model.to(device)

# Freeze ALL original parameters first
for param in exercise_model.parameters():
    param.requires_grad = False

# Replace specific layers with LoRA versions
# Like swapping out regular tools with adjustable ones
exercise_model.linear_relu_stack[0] = LoRALinear(
    exercise_model.linear_relu_stack[0], 
    rank=8,  # Low rank - like having 8 "adjustment knobs"
    alpha=16.0  # Scaling factor
)

exercise_model.linear_relu_stack[2] = LoRALinear(
    exercise_model.linear_relu_stack[2], 
    rank=8, 
    alpha=16.0
)

# Replace final layer (this one trains normally)
exercise_model.linear_relu_stack[4] = nn.Linear(512, 5)

def enable_lora_gradients(model):
    """Enable gradients only for LoRA parameters - like unlocking adjustment dials"""
    trainable_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name.lower() or 'linear_relu_stack.4' in name:
            param.requires_grad = True
            trainable_count += param.numel()
            print(f"Training enabled for: {name} (shape: {param.shape})")
    return trainable_count

trainable_count = enable_lora_gradients(exercise_model)

# Create optimizer with only trainable parameters
trainable_params = [p for p in exercise_model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=0.001, weight_decay=0.01)

total_params = sum(p.numel() for p in exercise_model.parameters())
print(f"\nParameter Summary:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_count:,}")
print(f"Trainable percentage: {100 * trainable_count / total_params:.2f}%")

def train_lora_only(model, train_loader, optimizer, criterion, epochs=5):
    """
    Train only the LoRA parameters
    Like fine-tuning specific skills while keeping core knowledge intact
    """
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass - original weights + LoRA adaptations
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass - only LoRA weights will be updated
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Example usage
criterion = nn.CrossEntropyLoss()

# Assuming you have a train_loader defined
# train_loader = DataLoader(your_dataset, batch_size=64, shuffle=True)

# Train only the LoRA parameters
print(f"\nStarting LoRA training...")
train_lora_only(exercise_model, train_loader, optimizer, criterion, epochs=10)

# Verify only LoRA parameters were trained
print("\nParameter update verification:")
for name, param in exercise_model.named_parameters():
    if param.requires_grad:
        print(f"✓ {name} - TRAINED (shape: {param.shape})")
    else:
        print(f"✗ {name} - FROZEN")

# Save only the LoRA weights (much smaller file!)
def save_lora_weights(model, path):
    """Save only the LoRA parameters - like saving just your custom adjustments"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name.lower() or 'linear_relu_stack.4' in name:
            lora_state[name] = param.data
    torch.save(lora_state, path)
    print(f"LoRA weights saved to {path}")

# Example: save_lora_weights(exercise_model, "lora_weights.pth")
