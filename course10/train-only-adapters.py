import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Your model setup (assuming Adapted class is defined)
exercise_model = NeuralNetwork()
exercise_model.to(device)

# Freeze all parameters first
for param in exercise_model.parameters():
    param.requires_grad = False

# Add adapters and modify final layer
exercise_model.linear_relu_stack[0] = Adapted(exercise_model.linear_relu_stack[0], bottleneck_size=30)
exercise_model.linear_relu_stack[2] = Adapted(exercise_model.linear_relu_stack[2], bottleneck_size=30)
exercise_model.linear_relu_stack[4] = nn.Linear(512, 5)

# Enable gradients ONLY for adapter parameters and final layer
def enable_adapter_gradients(model):
    """Enable gradients only for adapter layers - like unlocking specific doors in a building"""
    for name, param in model.named_parameters():
        if 'adapter' in name.lower() or 'linear_relu_stack.4' in name:
            param.requires_grad = True
            print(f"Training enabled for: {name}")

enable_adapter_gradients(exercise_model)

# Create optimizer with only trainable parameters
trainable_params = [p for p in exercise_model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=0.001)

print(f"Total parameters: {sum(p.numel() for p in exercise_model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

# Training loop
def train_adapter_only(model, train_loader, optimizer, criterion, epochs=5):
    """Train only the adapter layers - like fine-tuning a specific skill"""
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass - only adapter weights will be updated
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
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# Example usage
criterion = nn.CrossEntropyLoss()

# Assuming you have a train_loader defined
# train_loader = DataLoader(your_dataset, batch_size=64, shuffle=True)

# Train only the adapters
train_adapter_only(exercise_model, train_loader, optimizer, criterion, epochs=10)

# Verify only adapters were trained
print("\nParameter update verification:")
for name, param in exercise_model.named_parameters():
    if param.requires_grad:
        print(f"✓ {name} - TRAINED")
    else:
        print(f"✗ {name} - FROZEN")

