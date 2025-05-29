import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("SIGMOID SATURATION ANALYSIS")
print("="*60)

# ========================
# 1. WHAT IS SATURATION?
# ========================
print("\n1. UNDERSTANDING SATURATION")
print("-" * 40)

# Create input range
x = torch.linspace(-10, 10, 1000)

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Calculate sigmoid values
sigmoid_y = sigmoid(x)

# Calculate the derivative (gradient) of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # Chain rule: d/dx[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))

sigmoid_grad = sigmoid_derivative(x)

print("Sigmoid function: f(x) = 1 / (1 + e^(-x))")
print("Sigmoid derivative: f'(x) = f(x) * (1 - f(x))")

# ========================
# 2. NUMERICAL ANALYSIS OF SATURATION
# ========================
print(f"\n2. NUMERICAL EVIDENCE OF SATURATION")
print("-" * 45)

# Check specific values
test_points = [-10, -5, -2, 0, 2, 5, 10]
print(f"{'x':<6} {'sigmoid(x)':<12} {'gradient':<12} {'saturated?'}")
print("-" * 50)

for x_val in test_points:
    x_tensor = torch.tensor(float(x_val))
    sig_val = sigmoid(x_tensor).item()
    grad_val = sigmoid_derivative(x_tensor).item()
    
    # Consider "saturated" if gradient is very small
    is_saturated = grad_val < 0.01  # Less than 1% change rate
    status = "YES" if is_saturated else "NO"
    
    print(f"{x_val:<6} {sig_val:<12.6f} {grad_val:<12.6f} {status}")

# ========================
# 3. VISUAL ANALYSIS
# ========================
print(f"\n3. VISUAL ANALYSIS")
print("-" * 20)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Sigmoid function
ax1.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid')
ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Near 0 (saturated)')
ax1.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='Near 1 (saturated)')
ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Active region')

# Highlight saturation regions
ax1.fill_between(x, 0, sigmoid_y, where=(sigmoid_y < 0.01), alpha=0.3, color='red', label='Lower saturation')
ax1.fill_between(x, sigmoid_y, 1, where=(sigmoid_y > 0.99), alpha=0.3, color='red', label='Upper saturation')

ax1.set_xlabel('Input (x)')
ax1.set_ylabel('Sigmoid(x)')
ax1.set_title('Sigmoid Function - Showing Saturation Regions')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(-0.1, 1.1)

# Plot 2: Sigmoid gradient (derivative)
ax2.plot(x, sigmoid_grad, 'r-', linewidth=2, label='Sigmoid Gradient')
ax2.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Very small gradient')
ax2.axhline(y=0.25, color='g', linestyle='--', alpha=0.7, label='Maximum gradient')

# Highlight regions with small gradients
ax2.fill_between(x, 0, sigmoid_grad, where=(sigmoid_grad < 0.01), alpha=0.3, color='red', label='Vanishing gradient')

ax2.set_xlabel('Input (x)')
ax2.set_ylabel('Gradient')
ax2.set_title('Sigmoid Gradient - Shows Where Learning Slows Down')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# ========================
# 4. THE VANISHING GRADIENT PROBLEM
# ========================
print(f"\n4. THE VANISHING GRADIENT PROBLEM IN DEEP NETWORKS")
print("-" * 55)

print("In a deep network, gradients are multiplied through chain rule:")
print("Final gradient = grad_layer_n × grad_layer_(n-1) × ... × grad_layer_1")

# Simulate gradient flow through multiple layers
def simulate_gradient_flow(num_layers, input_value):
    """Simulate how gradients flow backwards through layers with sigmoid"""
    current_grad = 1.0  # Start with gradient of 1
    print(f"\nSimulating gradient flow for input x = {input_value}")
    print(f"Through {num_layers} layers with sigmoid activation:")
    print(f"{'Layer':<8} {'Local Grad':<12} {'Cumulative Grad':<15}")
    print("-" * 40)
    
    for layer in range(num_layers):
        # Each layer contributes its local gradient
        local_grad = sigmoid_derivative(torch.tensor(float(input_value))).item()
        current_grad *= local_grad
        print(f"{layer+1:<8} {local_grad:<12.6f} {current_grad:<15.8f}")
    
    return current_grad

# Test with saturated inputs
print("CASE 1: Input in saturated region (x = -5)")
final_grad_saturated = simulate_gradient_flow(5, -5)

print("\nCASE 2: Input in active region (x = 0)")  
final_grad_active = simulate_gradient_flow(5, 0)

print(f"\nComparison:")
print(f"Final gradient (saturated): {final_grad_saturated:.8f}")
print(f"Final gradient (active):    {final_grad_active:.8f}")
print(f"Ratio: {final_grad_active/final_grad_saturated:.2f}x stronger")

# ========================
# 5. COMPARISON WITH ReLU
# ========================
print(f"\n5. WHY ReLU SOLVES THE SATURATION PROBLEM")
print("-" * 48)

# ReLU function and derivative
def relu(x):
    return torch.maximum(torch.zeros_like(x), x)

def relu_derivative(x):
    return (x > 0).float()  # 1 if x > 0, else 0

# Compare gradients
x_comparison = torch.linspace(-5, 5, 100)
sigmoid_grad_comp = sigmoid_derivative(x_comparison)
relu_grad_comp = relu_derivative(x_comparison)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_comparison, sigmoid_grad_comp, 'r-', linewidth=2, label='Sigmoid Gradient')
plt.plot(x_comparison, relu_grad_comp, 'b-', linewidth=2, label='ReLU Gradient')
plt.xlabel('Input (x)')
plt.ylabel('Gradient')
plt.title('Gradient Comparison: Sigmoid vs ReLU')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_comparison, sigmoid(x_comparison), 'r-', linewidth=2, label='Sigmoid')
plt.plot(x_comparison, relu(x_comparison), 'b-', linewidth=2, label='ReLU')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.title('Function Comparison: Sigmoid vs ReLU')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================
# 6. QUANTITATIVE SATURATION METRICS
# ========================
print(f"\n6. QUANTITATIVE SATURATION METRICS")
print("-" * 42)

def analyze_saturation(func, derivative_func, x_range, threshold=0.01):
    """Analyze what percentage of the function is saturated"""
    y_values = func(x_range)
    grad_values = derivative_func(x_range)
    
    # Count saturated points (gradient below threshold)
    saturated_points = (grad_values < threshold).sum().item()
    total_points = len(x_range)
    saturation_percentage = (saturated_points / total_points) * 100
    
    # Find maximum gradient
    max_gradient = grad_values.max().item()
    
    return saturation_percentage, max_gradient

x_range = torch.linspace(-10, 10, 1000)

# Analyze sigmoid
sigmoid_saturation, sigmoid_max_grad = analyze_saturation(sigmoid, sigmoid_derivative, x_range)

# Analyze ReLU  
relu_saturation, relu_max_grad = analyze_saturation(relu, relu_derivative, x_range)

print(f"SIGMOID ANALYSIS:")
print(f"  Saturation percentage: {sigmoid_saturation:.1f}%")
print(f"  Maximum gradient: {sigmoid_max_grad:.4f}")

print(f"\nReLU ANALYSIS:")
print(f"  Saturation percentage: {relu_saturation:.1f}%")
print(f"  Maximum gradient: {relu_max_grad:.4f}")

print(f"\nKEY INSIGHTS:")
print(f"1. Sigmoid saturates over {sigmoid_saturation:.1f}% of its input range")
print(f"2. Maximum sigmoid gradient is only {sigmoid_max_grad:.4f}")
print(f"3. ReLU has gradient of 1.0 (when active) vs sigmoid's max of {sigmoid_max_grad:.4f}")
print(f"4. This is why deep networks with sigmoid struggle to learn!")

# ========================
# 7. PRACTICAL IMPLICATIONS
# ========================
print(f"\n7. PRACTICAL IMPLICATIONS")
print("-" * 30)

print("""
SATURATION DETECTION CRITERIA:

1. MATHEMATICAL:
   - Gradient < 0.01 means less than 1% change per unit input
   - Sigmoid saturates when |x| > 5 approximately
   - Function value very close to 0 or 1

2. VISUAL:
   - Nearly flat regions in the function plot
   - Near-zero regions in the gradient plot
   - Function asymptotically approaching limits

3. TRAINING SYMPTOMS:
   - Very slow learning in deep networks
   - Gradients that exponentially decay
   - Early layers barely updating

4. WHY THIS MATTERS:
   - Saturated neurons stop learning effectively
   - Information doesn't flow backwards (vanishing gradients)
   - Network gets "stuck" in poor solutions

SOLUTION: Use activation functions like ReLU that don't saturate!
""")

print("This mathematical analysis proves why ReLU revolutionized deep learning!")
