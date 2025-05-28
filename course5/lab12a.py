# Using Dropout in Regression - Fixed Version

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

# Create Data object
class Data(Dataset):
    def __init__(self, N_SAMPLES=40, noise_std=1, train=True):
        self.x = torch.linspace(-1, 1, N_SAMPLES).view(-1, 1)
        self.f = self.x ** 2
        self.len = N_SAMPLES  # BUG FIX: Added missing len attribute
        
        if train != True:
            torch.manual_seed(1)
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)
            torch.manual_seed(0)
        else:
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.x.numpy(), self.y.numpy(), label="Samples", alpha=0.7)
        plt.plot(self.x.numpy(), self.f.numpy(), label="True Function", color='orange', linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((-1, 1))
        plt.ylim((-2, 2.5))
        plt.legend(loc="best")
        plt.title("Training Data")
        plt.grid(True, alpha=0.3)
        plt.show()

# Create the datasets
print("Creating datasets...")
data_set = Data()
validation_set = Data(train=False)
print(f"Training samples: {len(data_set)}")
print(f"Validation samples: {len(validation_set)}")

# Plot the training data
data_set.plot()

# Neural Network with Dropout
class Net(nn.Module):
    def __init__(self, in_size, n_hidden, out_size, p=0):
        super(Net, self).__init__()
        self.dropout_prob = p
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)

    def forward(self, x):
        x = F.relu(self.drop(self.linear1(x)))
        x = F.relu(self.drop(self.linear2(x)))
        x = self.linear3(x)
        return x

# Create models
print("\nCreating models...")
model_no_dropout = Net(1, 300, 1, p=0)
model_with_dropout = Net(1, 300, 1, p=0.5)

print(f"Model without dropout: {sum(p.numel() for p in model_no_dropout.parameters())} parameters")
print(f"Model with dropout (p=0.5): {sum(p.numel() for p in model_with_dropout.parameters())} parameters")

# Set optimizers and loss function
optimizer_no_dropout = torch.optim.Adam(model_no_dropout.parameters(), lr=0.01)
optimizer_with_dropout = torch.optim.Adam(model_with_dropout.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Initialize loss tracking
LOSS = {
    'training_no_dropout': [],
    'validation_no_dropout': [],
    'training_with_dropout': [],
    'validation_with_dropout': []
}

epochs = 500
print_every = 100

def train_models(epochs):
    print(f"\nTraining both models for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Set models to training mode
        model_no_dropout.train()
        model_with_dropout.train()
        
        # Forward pass
        yhat_no_dropout = model_no_dropout(data_set.x)
        yhat_with_dropout = model_with_dropout(data_set.x)
        
        # Calculate training losses
        loss_no_dropout = criterion(yhat_no_dropout, data_set.y)
        loss_with_dropout = criterion(yhat_with_dropout, data_set.y)
        
        # Store training losses
        LOSS['training_no_dropout'].append(loss_no_dropout.item())
        LOSS['training_with_dropout'].append(loss_with_dropout.item())
        
        # Calculate validation losses (set models to eval mode)
        model_no_dropout.eval()
        model_with_dropout.eval()
        
        with torch.no_grad():  # No gradients needed for validation
            val_pred_no_dropout = model_no_dropout(validation_set.x)
            val_pred_with_dropout = model_with_dropout(validation_set.x)
            
            val_loss_no_dropout = criterion(val_pred_no_dropout, validation_set.y)
            val_loss_with_dropout = criterion(val_pred_with_dropout, validation_set.y)
            
            LOSS['validation_no_dropout'].append(val_loss_no_dropout.item())
            LOSS['validation_with_dropout'].append(val_loss_with_dropout.item())
        
        # Back to training mode for backpropagation
        model_no_dropout.train()
        model_with_dropout.train()
        
        # Backpropagation
        optimizer_no_dropout.zero_grad()
        optimizer_with_dropout.zero_grad()
        
        loss_no_dropout.backward()
        loss_with_dropout.backward()
        
        optimizer_no_dropout.step()
        optimizer_with_dropout.step()
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}")
            print(f"  No Dropout  - Train Loss: {loss_no_dropout.item():.6f}, Val Loss: {val_loss_no_dropout.item():.6f}")
            print(f"  With Dropout - Train Loss: {loss_with_dropout.item():.6f}, Val Loss: {val_loss_with_dropout.item():.6f}")
            print(f"  Gap (Train-Val): No Dropout = {loss_no_dropout.item() - val_loss_no_dropout.item():.6f}, "
                  f"With Dropout = {loss_with_dropout.item() - val_loss_with_dropout.item():.6f}")
            print("-" * 60)

# Train the models
train_models(epochs)

# Final evaluation
print("\nFinal Results:")
print("=" * 60)
final_train_no_drop = LOSS['training_no_dropout'][-1]
final_val_no_drop = LOSS['validation_no_dropout'][-1]
final_train_with_drop = LOSS['training_with_dropout'][-1]
final_val_with_drop = LOSS['validation_with_dropout'][-1]

print(f"No Dropout Model:")
print(f"  Final Training Loss:   {final_train_no_drop:.6f}")
print(f"  Final Validation Loss: {final_val_no_drop:.6f}")
print(f"  Overfitting Gap:       {final_train_no_drop - final_val_no_drop:.6f}")

print(f"\nWith Dropout Model:")
print(f"  Final Training Loss:   {final_train_with_drop:.6f}")
print(f"  Final Validation Loss: {final_val_with_drop:.6f}")
print(f"  Overfitting Gap:       {final_train_with_drop - final_val_with_drop:.6f}")

print(f"\nGeneralization Improvement: {final_val_no_drop - final_val_with_drop:.6f}")

# Set models to evaluation mode for predictions
model_no_dropout.eval()
model_with_dropout.eval()

# Make predictions
with torch.no_grad():
    yhat_no_dropout = model_no_dropout(data_set.x)
    yhat_with_dropout = model_with_dropout(data_set.x)
    
    # Also predict on validation set
    val_pred_no_dropout = model_no_dropout(validation_set.x)
    val_pred_with_dropout = model_with_dropout(validation_set.x)

# Plot predictions on training data
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(data_set.x.numpy(), data_set.y.numpy(), label="Training Samples", alpha=0.7, s=30)
plt.plot(data_set.x.numpy(), data_set.f.numpy(), label="True Function", color='orange', linewidth=2)
plt.plot(data_set.x.numpy(), yhat_no_dropout.numpy(), label='No Dropout', color='red', linewidth=2)
plt.plot(data_set.x.numpy(), yhat_with_dropout.numpy(), label="With Dropout", color='green', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-1, 1))
plt.ylim((-2, 2.5))
plt.legend()
plt.title("Predictions on Training Data")
plt.grid(True, alpha=0.3)

# Plot predictions on validation data
plt.subplot(2, 2, 2)
plt.scatter(validation_set.x.numpy(), validation_set.y.numpy(), label="Validation Samples", alpha=0.7, s=30, color='purple')
plt.plot(validation_set.x.numpy(), validation_set.f.numpy(), label="True Function", color='orange', linewidth=2)
plt.plot(validation_set.x.numpy(), val_pred_no_dropout.numpy(), label='No Dropout', color='red', linewidth=2)
plt.plot(validation_set.x.numpy(), val_pred_with_dropout.numpy(), label="With Dropout", color='green', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-1, 1))
plt.ylim((-2, 2.5))
plt.legend()
plt.title("Predictions on Validation Data")
plt.grid(True, alpha=0.3)

# Plot training curves
plt.subplot(2, 1, 2)
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, LOSS['training_no_dropout'], label='Training (No Dropout)', color='red', alpha=0.7)
plt.plot(epochs_range, LOSS['validation_no_dropout'], label='Validation (No Dropout)', color='red', linestyle='--')
plt.plot(epochs_range, LOSS['training_with_dropout'], label='Training (With Dropout)', color='green', alpha=0.7)
plt.plot(epochs_range, LOSS['validation_with_dropout'], label='Validation (With Dropout)', color='green', linestyle='--')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to better see the differences

plt.tight_layout()
plt.show()

# LESSON SUMMARY
print("\n" + "="*80)
print("DROPOUT LESSON SUMMARY")
print("="*80)

print("\n🎯 KEY OBSERVATIONS:")
print(f"1. WITHOUT DROPOUT:")
print(f"   - Lower training loss ({final_train_no_drop:.6f}) but higher validation loss ({final_val_no_drop:.6f})")
print(f"   - Large gap = OVERFITTING (memorizing training data)")

print(f"\n2. WITH DROPOUT:")
print(f"   - Higher training loss ({final_train_with_drop:.6f}) but lower validation loss ({final_val_with_drop:.6f})")
print(f"   - Smaller gap = BETTER GENERALIZATION")

print(f"\n📊 ANALOGY - Dropout is like studying with distractions:")
print(f"   - Without dropout: Like studying in perfect silence (overfits to ideal conditions)")
print(f"   - With dropout: Like studying with background noise (learns robust patterns)")

print(f"\n🔄 WHAT TO DO IN PRACTICE:")
print(f"   IF your model shows:")
print(f"   • Training loss << Validation loss → ADD/INCREASE dropout")
print(f"   • Training loss ≈ Validation loss → Current dropout is good")
print(f"   • Training loss >> Validation loss → REDUCE dropout (underfitting)")

print(f"\n💡 REMEMBER:")
print(f"   - Use .train() mode during training (dropout active)")
print(f"   - Use .eval() mode during inference (dropout disabled)")
print(f"   - Dropout helps prevent overfitting by forcing the network to not rely on specific neurons")

print("\n" + "="*80)
