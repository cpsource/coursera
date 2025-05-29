# Batch Normalization with the MNIST Dataset - Fixed Version

# These are the libraries will be used for this lab.
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

print("Libraries loaded successfully!")
print(f"PyTorch version: {torch.__version__}")

# Define the Neural Network Model using Batch Normalization

class NetBatchNorm(nn.Module):

    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):
        super(NetBatchNorm, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.bn2 = nn.BatchNorm1d(n_hidden2)

    # Prediction
    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.linear1(x)))
        x = self.bn2(torch.sigmoid(self.linear2(x)))
        x = self.linear3(x)
        return x

    # Activations, to analyze results
    def activation(self, x):
        out = []
        z1 = self.bn1(self.linear1(x))
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().numpy().reshape(-1))  # Fixed: removed duplicate reshape
        z2 = self.bn2(self.linear2(a1))
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out

# Class Net for Neural Network Model

class Net(nn.Module):

    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):

        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x

    # Activations, to analyze results
    def activation(self, x):
        out = []
        z1 = self.linear1(x)
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().numpy().reshape(-1))  # Fixed: removed duplicate reshape
        z2 = self.linear2(a1)
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out

print("Neural network classes defined successfully!")

# Define the function to train model

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    print(f"Starting training for {epochs} epochs...")
    i = 0
    useful_stuff = {'training_loss':[], 'validation_accuracy':[]}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses = []
        
        for i, (x, y) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.data.item())
            useful_stuff['training_loss'].append(loss.data.item())
        
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"  Average training loss: {avg_epoch_loss:.4f}")

        # Validation accuracy calculation
        correct = 0
        total = 0  # Fixed: added total counter
        model.eval()
        with torch.no_grad():  # Added: no_grad for efficiency
            for x, y in validation_loader:
                yhat = model(x.view(-1, 28 * 28))
                _, label = torch.max(yhat, 1)
                correct += (label == y).sum().item()
                total += y.size(0)  # Fixed: count actual samples

        accuracy = 100 * (correct / total)  # Fixed: use actual total
        useful_stuff['validation_accuracy'].append(accuracy)
        print(f"  Validation accuracy: {accuracy:.2f}% ({correct}/{total})")

    return useful_stuff

# Make Some Data

print("\nLoading MNIST dataset...")

# load the train dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
print(f"Training dataset size: {len(train_dataset)}")

# load the validation dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
print(f"Validation dataset size: {len(validation_dataset)}")

# Create Data Loader for both train and validating
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(validation_loader)}")

# Define Neural Network, Criterion function, Optimizer and Train the Model

# Create the criterion function
criterion = nn.CrossEntropyLoss()
print("Loss function: CrossEntropyLoss")

# Set the parameters
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10

print(f"\nNetwork architecture:")
print(f"  Input dimension: {input_dim}")
print(f"  Hidden layers: {hidden_dim} neurons each")
print(f"  Output dimension: {output_dim}")

# Create model with Batch Normalization, optimizer and train the model
print("\n" + "="*50)
print("TRAINING MODEL WITH BATCH NORMALIZATION")
print("="*50)

model_norm = NetBatchNorm(input_dim, hidden_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model_norm.parameters(), lr=0.1)
training_results_Norm = train(model_norm, criterion, train_loader, validation_loader, optimizer, epochs=5)

# Create model without Batch Normalization, optimizer and train the model
print("\n" + "="*50)
print("TRAINING MODEL WITHOUT BATCH NORMALIZATION")
print("="*50)

model = Net(input_dim, hidden_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=5)

# Analyze Results
print("\n" + "="*50)
print("ANALYZING RESULTS")
print("="*50)

model.eval()
model_norm.eval()

print("Generating activation histograms...")
out = model.activation(validation_dataset[0][0].reshape(-1, 28*28))
plt.figure(figsize=(10, 6))
plt.hist(out[2], alpha=0.7, label='Model without Batch Normalization', bins=30)
out_norm = model_norm.activation(validation_dataset[0][0].reshape(-1, 28*28))
plt.hist(out_norm[2], alpha=0.7, label='Model with Batch Normalization', bins=30)
plt.xlabel("Activation Values")
plt.ylabel("Frequency")
plt.title("Distribution of Activations in Second Hidden Layer")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compare the training loss for each iteration
print("Plotting training loss comparison...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_results['training_loss'], label='No Batch Normalization', alpha=0.8)
plt.plot(training_results_Norm['training_loss'], label='Batch Normalization', alpha=0.8)
plt.ylabel('Training Loss')
plt.xlabel('Iterations')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot the diagram to show the accuracy
plt.subplot(1, 2, 2)
plt.plot(training_results['validation_accuracy'], label='No Batch Normalization', marker='o')
plt.plot(training_results_Norm['validation_accuracy'], label='Batch Normalization', marker='s')
plt.ylabel('Validation Accuracy (%)')
plt.xlabel('Epochs')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final results summary
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

final_acc_no_bn = training_results['validation_accuracy'][-1]
final_acc_with_bn = training_results_Norm['validation_accuracy'][-1]
final_loss_no_bn = training_results['training_loss'][-1]
final_loss_with_bn = training_results_Norm['training_loss'][-1]

print(f"Model WITHOUT Batch Normalization:")
print(f"  Final validation accuracy: {final_acc_no_bn:.2f}%")
print(f"  Final training loss: {final_loss_no_bn:.4f}")

print(f"\nModel WITH Batch Normalization:")
print(f"  Final validation accuracy: {final_acc_with_bn:.2f}%")
print(f"  Final training loss: {final_loss_with_bn:.4f}")

print(f"\nImprovement with Batch Normalization:")
print(f"  Accuracy improvement: {final_acc_with_bn - final_acc_no_bn:.2f} percentage points")
print(f"  Loss reduction: {final_loss_no_bn - final_loss_with_bn:.4f}")

if final_acc_with_bn > final_acc_no_bn:
    print("✓ Batch Normalization improved model performance!")
else:
    print("⚠ Batch Normalization did not improve performance in this case.")
