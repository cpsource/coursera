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

# Define activation functions
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh_func(x):
    return torch.tanh(x)

# Calculate function values
sigmoid_y = sigmoid(x)
tanh_y = tanh_func(x)

# Calculate the derivatives (gradients)
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # Chain rule: d/dx[sigmoid(x)] = sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    t = torch.tanh(x)
    return 1 - t**2  # d/dx[tanh(x)] = 1 - tanh²(x)

sigmoid_grad = sigmoid_derivative(x)
tanh_grad = tanh_derivative(x)

print("Sigmoid function: f(x) = 1 / (1 + e^(-x))")
print("Sigmoid derivative: f'(x) = f(x) * (1 - f(x))")
print("Tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))")
print("Tanh derivative: f'(x) = 1 - tanh²(x)")

# ========================
# 2. NUMERICAL ANALYSIS OF SATURATION
# ========================
print(f"\n2. NUMERICAL EVIDENCE OF SATURATION")
print("-" * 45)

# Check specific values
test_points = [-10, -5, -2, 0, 2, 5, 10]
print(f"{'x':<6} {'sigmoid(x)':<12} {'sig_grad':<12} {'tanh(x)':<12} {'tanh_grad':<12} {'Saturated?'}")
print("-" * 80)

for x_val in test_points:
    x_tensor = torch.tensor(float(x_val))
    sig_val = sigmoid(x_tensor).item()
    sig_grad_val = sigmoid_derivative(x_tensor).item()
    tanh_val = tanh_func(x_tensor).item()
    tanh_grad_val = tanh_derivative(x_tensor).item()
    
    # Consider "saturated" if gradient is very small for both functions
    sig_saturated = sig_grad_val < 0.01
    tanh_saturated = tanh_grad_val < 0.01
    
    status = f"Sig:{('Y' if sig_saturated else 'N')} Tanh:{('Y' if tanh_saturated else 'N')}"
    
    print(f"{x_val:<6} {sig_val:<12.6f} {sig_grad_val:<12.6f} {tanh_val:<12.6f} {tanh_grad_val:<12.6f} {status}")

print(f"\nKey Observations:")
print(f"- Sigmoid range: [0, 1], Tanh range: [-1, 1]")
print(f"- Sigmoid max gradient: ~0.25, Tanh max gradient: 1.0")
print(f"- Tanh is zero-centered (better for training)")
print(f"- Both saturate at extreme values, but tanh less severely")

# ========================
# 3. VISUAL ANALYSIS
# ========================
print(f"\n3. VISUAL ANALYSIS")
print("-" * 20)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Function comparison
ax1.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid')
ax1.plot(x, tanh_y, 'g-', linewidth=2, label='Tanh')
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axhline(y=0.5, color='b', linestyle='--', alpha=0.5, label='Sigmoid center')
ax1.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Tanh center')

ax1.set_xlabel('Input (x)')
ax1.set_ylabel('Output')
ax1.set_title('Function Comparison: Sigmoid vs Tanh')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(-1.2, 1.2)

# Plot 2: Gradient comparison
ax2.plot(x, sigmoid_grad, 'b-', linewidth=2, label='Sigmoid Gradient')
ax2.plot(x, tanh_grad, 'g-', linewidth=2, label='Tanh Gradient')
ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Saturation threshold')
ax2.axhline(y=0.25, color='b', linestyle=':', alpha=0.7, label='Sigmoid max (0.25)')
ax2.axhline(y=1.0, color='g', linestyle=':', alpha=0.7, label='Tanh max (1.0)')

ax2.set_xlabel('Input (x)')
ax2.set_ylabel('Gradient')
ax2.set_title('Gradient Comparison: Sigmoid vs Tanh')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Sigmoid saturation regions
ax3.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid')
ax3.fill_between(x, 0, sigmoid_y, where=(sigmoid_grad < 0.01), alpha=0.3, color='red', label='Saturated regions')
ax3.axhline(y=0.01, color='r', linestyle='--', alpha=0.7)
ax3.axhline(y=0.99, color='r', linestyle='--', alpha=0.7)

ax3.set_xlabel('Input (x)')
ax3.set_ylabel('Sigmoid(x)')
ax3.set_title('Sigmoid Saturation Regions')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(-0.1, 1.1)

# Plot 4: Tanh saturation regions
ax4.plot(x, tanh_y, 'g-', linewidth=2, label='Tanh')
ax4.fill_between(x, -1, tanh_y, where=(tanh_grad < 0.01), alpha=0.3, color='red', label='Saturated regions')
ax4.axhline(y=-0.99, color='r', linestyle='--', alpha=0.7, label='Saturation boundaries')
ax4.axhline(y=0.99, color='r', linestyle='--', alpha=0.7)

ax4.set_xlabel('Input (x)')
ax4.set_ylabel('Tanh(x)')
ax4.set_title('Tanh Saturation Regions')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_ylim(-1.2, 1.2)

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
def simulate_gradient_flow(num_layers, input_value, activation_type='sigmoid'):
    """Simulate how gradients flow backwards through layers"""
    current_grad = 1.0  # Start with gradient of 1
    print(f"\nSimulating gradient flow for input x = {input_value}")
    print(f"Through {num_layers} layers with {activation_type} activation:")
    print(f"{'Layer':<8} {'Local Grad':<12} {'Cumulative Grad':<15}")
    print("-" * 40)
    
    for layer in range(num_layers):
        # Each layer contributes its local gradient
        if activation_type == 'sigmoid':
            local_grad = sigmoid_derivative(torch.tensor(float(input_value))).item()
        elif activation_type == 'tanh':
            local_grad = tanh_derivative(torch.tensor(float(input_value))).item()
        else:
            raise ValueError("activation_type must be 'sigmoid' or 'tanh'")
            
        current_grad *= local_grad
        print(f"{layer+1:<8} {local_grad:<12.6f} {current_grad:<15.8f}")
    
    return current_grad

# Test with saturated inputs for both activations
print("CASE 1: Input in saturated region (x = -5)")
print("Sigmoid:")
final_grad_sigmoid_sat = simulate_gradient_flow(5, -5, 'sigmoid')
print("Tanh:")
final_grad_tanh_sat = simulate_gradient_flow(5, -5, 'tanh')

print("\nCASE 2: Input in active region (x = 0)")  
print("Sigmoid:")
final_grad_sigmoid_active = simulate_gradient_flow(5, 0, 'sigmoid')
print("Tanh:")
final_grad_tanh_active = simulate_gradient_flow(5, 0, 'tanh')

print(f"\nComparison:")
print(f"Final gradient - Sigmoid (saturated): {final_grad_sigmoid_sat:.8f}")
print(f"Final gradient - Tanh (saturated):    {final_grad_tanh_sat:.8f}")
print(f"Final gradient - Sigmoid (active):    {final_grad_sigmoid_active:.8f}")
print(f"Final gradient - Tanh (active):       {final_grad_tanh_active:.8f}")
print(f"\nTanh vs Sigmoid advantage:")
print(f"- In saturated region: {final_grad_tanh_sat/final_grad_sigmoid_sat:.2f}x stronger")
print(f"- In active region: {final_grad_tanh_active/final_grad_sigmoid_active:.2f}x stronger")

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
tanh_grad_comp = tanh_derivative(x_comparison)
relu_grad_comp = relu_derivative(x_comparison)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.plot(x_comparison, sigmoid_grad_comp, 'b-', linewidth=2, label='Sigmoid Gradient')
plt.plot(x_comparison, tanh_grad_comp, 'g-', linewidth=2, label='Tanh Gradient')
plt.plot(x_comparison, relu_grad_comp, 'r-', linewidth=2, label='ReLU Gradient')
plt.xlabel('Input (x)')
plt.ylabel('Gradient')
plt.title('Gradient Comparison: All Three')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x_comparison, sigmoid(x_comparison), 'b-', linewidth=2, label='Sigmoid')
plt.plot(x_comparison, tanh_func(x_comparison), 'g-', linewidth=2, label='Tanh')
plt.plot(x_comparison, relu(x_comparison), 'r-', linewidth=2, label='ReLU')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.title('Function Comparison: All Three')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Show the "active" regions (where gradient > 0.1)
sigmoid_active = (sigmoid_grad_comp > 0.1).float()
tanh_active = (tanh_grad_comp > 0.1).float()
relu_active = (relu_grad_comp > 0.1).float()

plt.plot(x_comparison, sigmoid_active, 'b-', linewidth=3, label='Sigmoid Active Region')
plt.plot(x_comparison, tanh_active + 0.1, 'g-', linewidth=3, label='Tanh Active Region')
plt.plot(x_comparison, relu_active + 0.2, 'r-', linewidth=3, label='ReLU Active Region')
plt.xlabel('Input (x)')
plt.ylabel('Active (gradient > 0.1)')
plt.title('Active Learning Regions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.4)

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

# Analyze tanh
tanh_saturation, tanh_max_grad = analyze_saturation(tanh_func, tanh_derivative, x_range)

# Analyze ReLU  
relu_saturation, relu_max_grad = analyze_saturation(relu, relu_derivative, x_range)

print(f"SIGMOID ANALYSIS:")
print(f"  Saturation percentage: {sigmoid_saturation:.1f}%")
print(f"  Maximum gradient: {sigmoid_max_grad:.4f}")
print(f"  Range: [0, 1]")
print(f"  Zero-centered: No")

print(f"\nTANH ANALYSIS:")
print(f"  Saturation percentage: {tanh_saturation:.1f}%")
print(f"  Maximum gradient: {tanh_max_grad:.4f}")
print(f"  Range: [-1, 1]")
print(f"  Zero-centered: Yes")

print(f"\nReLU ANALYSIS:")
print(f"  Saturation percentage: {relu_saturation:.1f}%")
print(f"  Maximum gradient: {relu_max_grad:.4f}")
print(f"  Range: [0, ∞]")
print(f"  Zero-centered: No")

print(f"\nCOMPARATIVE INSIGHTS:")
print(f"1. Tanh has {tanh_max_grad/sigmoid_max_grad:.1f}x stronger maximum gradient than sigmoid")
print(f"2. Both sigmoid and tanh saturate over ~{sigmoid_saturation:.0f}% of input range")
print(f"3. Tanh is zero-centered (mean activation ≈ 0), sigmoid is not (mean ≈ 0.5)")
print(f"4. ReLU doesn't saturate in the positive region")
print(f"5. Zero-centering helps with gradient flow and convergence")

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
   - Tanh saturates when |x| > 3 approximately
   - Function value very close to asymptotic limits

2. VISUAL:
   - Nearly flat regions in the function plot
   - Near-zero regions in the gradient plot
   - Function asymptotically approaching limits

3. TRAINING SYMPTOMS:
   - Very slow learning in deep networks
   - Gradients that exponentially decay
   - Early layers barely updating

4. SIGMOID vs TANH vs ReLU COMPARISON:

   SIGMOID:
   ✗ Range [0,1] - not zero-centered
   ✗ Max gradient only 0.25
   ✗ Saturates heavily
   ✗ Vanishing gradient problem
   
   TANH:
   ✓ Range [-1,1] - zero-centered (better!)
   ✓ Max gradient 1.0 (4x better than sigmoid)
   ✗ Still saturates
   ✗ Still has vanishing gradient problem (but less severe)
   
   ReLU:
   ✓ No saturation in positive region
   ✓ Gradient 1.0 when active
   ✓ Fast computation
   ✓ Sparse activations
   ✗ Not zero-centered
   ✗ "Dying ReLU" problem

5. WHY TANH IS BETTER THAN SIGMOID:
   - Zero-centered activations help gradient flow
   - 4x stronger maximum gradient (1.0 vs 0.25)
   - Better for traditional neural networks (pre-ReLU era)
   - Still used in LSTMs and some specific contexts

6. WHY ReLU REVOLUTIONIZED DEEP LEARNING:
   - Eliminates vanishing gradient problem
   - Enables training of very deep networks
   - Computationally efficient
   - Simple yet effective

HISTORICAL NOTE: 
Tanh was the preferred activation before ReLU, because it was clearly 
superior to sigmoid. ReLU then revolutionized the field by solving 
the fundamental saturation problem entirely (in the positive region).
""")

print("This analysis shows the evolution: Sigmoid → Tanh → ReLU → Modern variants!")
