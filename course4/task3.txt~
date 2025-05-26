# League of Legends Match Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Step 1: Load dataset
data = pd.read_csv("league_of_legends_data_large.csv")  # adjust path as needed

# Step 2: Split into features and target
X = data.drop('win', axis=1)
y = data['win']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# ✅ Check shapes
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_test :", X_test_tensor.shape)
print("y_test :", y_test_tensor.shape)

# Task 2: Implement a logistic regression model using PyTorch.

print("Exercise 2")

import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Step 2: Initialize model, loss, and optimizer

# Replace this with X_train_tensor.shape[1] if you've loaded data already
#input_dim = 10  # e.g., 10 features in your dataset
input_dim = X_train_tensor.shape[1]

model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Optional: Print model summary
print(model)

# Step 3: Model Training

print("Exercise 3")

import torch

# Set number of epochs
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    model.train()  # Training mode

    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass: predict y from X
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    # Predictions
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    # Apply threshold of 0.5 to convert probabilities to binary predictions
    train_preds_class = (train_preds >= 0.5).float()
    test_preds_class = (test_preds >= 0.5).float()

    # Accuracy calculation
    train_accuracy = (train_preds_class == y_train_tensor).float().mean().item()
    test_accuracy = (test_preds_class == y_test_tensor).float().mean().item()

    print(f"\n✅ Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"✅ Test Accuracy    : {test_accuracy * 100:.2f}%")

print("Exercise 4")

import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Redefine the model (fresh weights)
input_dim = X_train_tensor.shape[1]

model = LogisticRegressionModel(input_dim)

# Step 2: Optimizer with L2 regularization (weight decay)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# Step 3: Binary Cross-Entropy Loss
criterion = nn.BCELoss()

# Step 4: Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Step 5: Evaluation
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    # Threshold at 0.5 to classify
    train_preds_class = (train_preds >= 0.5).float()
    test_preds_class = (test_preds >= 0.5).float()

    # Accuracy
    train_acc = (train_preds_class == y_train_tensor).float().mean().item()
    test_acc = (test_preds_class == y_test_tensor).float().mean().item()

print(f"\n✅ Training Accuracy with L2 Regularization: {train_acc * 100:.2f}%")
print(f"✅ Test Accuracy with L2 Regularization    : {test_acc * 100:.2f}%")

# Exercise 5

print("Exercise 5")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import numpy as np

# Convert predictions and labels to NumPy for sklearn compatibility
y_true = y_test_tensor.numpy()
y_probs = test_preds.numpy()           # predicted probabilities
y_pred_class = test_preds_class.numpy()  # predicted classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(ticks=[0.5, 1.5], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0.5, 1.5], labels=['Class 0', 'Class 1'], rotation=0)
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Classification Report
print("📋 Classification Report:")
print(classification_report(y_true, y_pred_class, target_names=["Class 0", "Class 1"]))

# Exercise 6

print("Exercise 6")

import torch

# 🔹 Step 1: Save the trained model's state dictionary
torch.save(model.state_dict(), 'logistic_model.pth')
print("✅ Model saved as logistic_model.pth")

# 🔹 Step 2: Reload the model from saved weights
# Get input dimension from training data
input_dim = X_test_tensor.shape[1]
loaded_model = LogisticRegressionModel(input_dim)

# Load saved parameters into the new model
loaded_model.load_state_dict(torch.load('logistic_model.pth'))
print("✅ Model loaded from logistic_model.pth")

# 🔹 Step 3: Evaluate the loaded model
loaded_model.eval()
with torch.no_grad():
    test_preds = loaded_model(X_test_tensor)
    test_preds_class = (test_preds >= 0.5).float()
    test_accuracy = (test_preds_class == y_test_tensor).float().mean().item()

print(f"📊 Test Accuracy (loaded model): {test_accuracy * 100:.2f}%")

# Exercise 7

print("Exercise 7")

import torch
import torch.nn as nn
import torch.optim as optim

# Learning rates to test
learning_rates = [0.01, 0.05, 0.1]

# Number of epochs for each test
num_epochs = 100

# Track test accuracies
accuracy_results = {}

# Input dimension from dataset
input_dim = X_train_tensor.shape[1]

for lr in learning_rates:
    # Reinitialize model and optimizer for each learning rate
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        test_preds_class = (test_preds >= 0.5).float()
        test_accuracy = (test_preds_class == y_test_tensor).float().mean().item()
        accuracy_results[lr] = test_accuracy

    print(f"Learning Rate: {lr} → Test Accuracy: {test_accuracy * 100:.2f}%")

# Identify the best learning rate
best_lr = max(accuracy_results, key=accuracy_results.get)
print(f"\n✅ Best Learning Rate: {best_lr} with Test Accuracy: {accuracy_results[best_lr] * 100:.2f}%")

# Exercise 8

print("Exercise 8")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Extract weights from the trained model
# Assumes model is an instance of LogisticRegressionModel
weights = model.linear.weight.data.numpy().flatten()

# Step 2: Get the original feature names
feature_names = X.columns.tolist()

# Step 3: Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': weights,
    'AbsWeight': np.abs(weights)
})

# Step 4: Sort by absolute weight to find most influential features
importance_df = importance_df.sort_values(by='AbsWeight', ascending=True)

# Step 5: Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Weight'], color='skyblue')
plt.axvline(x=0, color='gray', linestyle='--')
plt.title("Feature Importance (Logistic Regression Weights)")
plt.xlabel("Weight (Impact on Prediction)")
plt.tight_layout()
plt.show()

# Optional: print the sorted table
print("\n📋 Sorted Feature Importances:")
print(importance_df.sort_values(by='AbsWeight', ascending=False).round(4))


