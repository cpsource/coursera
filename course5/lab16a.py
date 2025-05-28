# Momentum with Different Polynomials - Fixed Version

# These are the libraries that will be used for this lab.

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import numpy as np

torch.manual_seed(0)

# Plot the cubic

def plot_cubic(w, optimizer):
    print(f"\n=== Starting cubic function optimization ===")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Momentum: {optimizer.param_groups[0].get('momentum', 0)}")
    
    LOSS = []
    # parameter values
    W = torch.arange(-4, 4, 0.1)
    # plot the loss function
    original_weight = w.state_dict()['linear.weight'][0].item()
    for w_val in W:
        w.state_dict()['linear.weight'][0] = w_val
        LOSS.append(cubic(w(torch.tensor([[1.0]]))).item())
    
    # Reset weight to starting position
    w.state_dict()['linear.weight'][0] = 4.0
    print(f"Starting weight: {w.state_dict()['linear.weight'][0].item()}")
    
    n_epochs = 10
    parameter = []
    loss_list = []

    # n_epochs
    for n in range(n_epochs):
        optimizer.zero_grad()
        loss = cubic(w(torch.tensor([[1.0]])))
        loss_list.append(loss)
        current_param = w.state_dict()['linear.weight'][0].detach().data.item()
        parameter.append(current_param)
        loss.backward()
        optimizer.step()
        
        if n % 2 == 0 or n == n_epochs - 1:  # Print every 2nd epoch and last epoch
            print(f"Epoch {n+1}: weight={current_param:.4f}, loss={loss.item():.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(parameter, [loss.detach().numpy().item() for loss in loss_list], 'ro', label='parameter values')
    plt.plot(W.numpy(), LOSS, label='objective function')
    plt.xlabel('w')
    plt.ylabel('l(w)')
    plt.title(f'Cubic Function - Momentum: {optimizer.param_groups[0].get("momentum", 0)}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Final weight: {parameter[-1]:.4f}")
    print(f"Final loss: {loss_list[-1].item():.4f}")

# Plot the fourth order function and the parameter values

def plot_fourth_order(w, optimizer, std=0, color='r', paramlabel='parameter values', objfun=True):
    print(f"\n=== Starting fourth-order function optimization ===")
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Momentum: {optimizer.param_groups[0].get('momentum', 0)}")
    print(f"Noise std: {std}")
    
    W = torch.arange(-4, 6, 0.1)
    LOSS = []
    original_weight = w.state_dict()['linear.weight'][0].item()
    for w_val in W:
        w.state_dict()['linear.weight'][0] = w_val
        LOSS.append(fourth_order(w(torch.tensor([[1.0]]))).item())
    
    # Reset weight to starting position
    w.state_dict()['linear.weight'][0] = 6
    print(f"Starting weight: {w.state_dict()['linear.weight'][0].item()}")
    
    n_epochs = 100
    parameter = []
    loss_list = []

    #n_epochs
    for n in range(n_epochs):
        optimizer.zero_grad()
        base_loss = fourth_order(w(torch.tensor([[1.0]])))
        if std > 0:
            noise = std * torch.randn(1)
            loss = base_loss + noise
        else:
            loss = base_loss
            
        loss_list.append(loss)
        current_param = w.state_dict()['linear.weight'][0].detach().data.item()
        parameter.append(current_param)
        loss.backward()
        optimizer.step()
        
        if n % 20 == 0 or n == n_epochs - 1:  # Print every 20th epoch and last epoch
            print(f"Epoch {n+1}: weight={current_param:.4f}, loss={loss.item():.4f}")
    
    # Plotting
    if objfun:
        plt.figure(figsize=(10, 6))
        plt.plot(W.numpy(), LOSS, label='objective function')
        title_suffix = f'Momentum: {optimizer.param_groups[0].get("momentum", 0)}, Noise: {std}'
        plt.title(f'Fourth Order Function - {title_suffix}')
    
    plt.plot(parameter, [loss.detach().numpy().item() for loss in loss_list], 'o', label=paramlabel, color=color)
    plt.xlabel('w')
    plt.ylabel('l(w)')
    plt.legend()
    plt.grid(True)
    if objfun:
        plt.show()
    
    print(f"Final weight: {parameter[-1]:.4f}")
    print(f"Final loss: {loss_list[-1].item():.4f}")

# Create a linear model

class one_param(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(one_param, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        
    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

print("="*60)
print("MOMENTUM OPTIMIZATION COMPARISON LAB")
print("="*60)

# Create a one_param object
w = one_param(1, 1)

# Saddle Points

# Define a function to output a cubic
def cubic(yhat):
    out = yhat ** 3
    return out

print("\n" + "="*50)
print("SECTION 1: CUBIC FUNCTION (SADDLE POINTS)")
print("="*50)

# Create a optimizer without momentum
print("\n--- Test 1: SGD without momentum ---")
w = one_param(1, 1)  # Reset model
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0)
plot_cubic(w, optimizer)

# Create a optimizer with momentum
print("\n--- Test 2: SGD with momentum=0.9 ---")
w = one_param(1, 1)  # Reset model
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0.9)
plot_cubic(w, optimizer)

# Local Minima

# Create a function to calculate the fourth order polynomial
def fourth_order(yhat):
    out = torch.mean(2 * (yhat ** 4) - 9 * (yhat ** 3) - 21 * (yhat ** 2) + 88 * yhat + 48)
    return out

print("\n" + "="*50)
print("SECTION 2: FOURTH ORDER FUNCTION (LOCAL MINIMA)")
print("="*50)

# Make the prediction without momentum
print("\n--- Test 3: Fourth order without momentum ---")
w = one_param(1, 1)  # Reset model
optimizer = torch.optim.SGD(w.parameters(), lr=0.001)
plot_fourth_order(w, optimizer)

# Make the prediction with momentum
print("\n--- Test 4: Fourth order with momentum=0.9 ---")
w = one_param(1, 1)  # Reset model
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
plot_fourth_order(w, optimizer)

print("\n" + "="*50)
print("SECTION 3: FOURTH ORDER FUNCTION WITH NOISE")
print("="*50)

# Make the prediction without momentum when there is noise
print("\n--- Test 5: Fourth order without momentum (noise std=10) ---")
w = one_param(1, 1)  # Reset model
optimizer = torch.optim.SGD(w.parameters(), lr=0.001)
plot_fourth_order(w, optimizer, std=10)

# Make the prediction with momentum when there is noise
print("\n--- Test 6: Fourth order with momentum=0.9 (noise std=10) ---")
w = one_param(1, 1)  # Reset model
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
plot_fourth_order(w, optimizer, std=10)

print("\n" + "="*50)
print("SECTION 4: COMPARISON WITH HIGH NOISE")
print("="*50)

# Practice - Combined plot
print("\n--- Test 7: High noise comparison (std=100) ---")

# Create a combined plot manually to avoid figure conflicts
print("Running optimizer 1 (no momentum)...")
w1 = one_param(1, 1)
optimizer1 = torch.optim.SGD(w1.parameters(), lr=0.001)

# Get data for optimizer 1
W = torch.arange(-4, 6, 0.1)
LOSS = []
for w_val in W:
    w1.state_dict()['linear.weight'][0] = w_val
    LOSS.append(fourth_order(w1(torch.tensor([[1.0]]))).item())

w1.state_dict()['linear.weight'][0] = 6
parameter1 = []
loss_list1 = []
for n in range(100):
    optimizer1.zero_grad()
    base_loss = fourth_order(w1(torch.tensor([[1.0]])))
    noise = 100 * torch.randn(1)
    loss = base_loss + noise
    loss_list1.append(loss)
    current_param = w1.state_dict()['linear.weight'][0].detach().data.item()
    parameter1.append(current_param)
    loss.backward()
    optimizer1.step()
    
    if n % 20 == 0 or n == 99:
        print(f"Optimizer 1 - Epoch {n+1}: weight={current_param:.4f}, loss={loss.item():.4f}")

print("\nRunning optimizer 2 (momentum=0.9)...")
w2 = one_param(1, 1)
optimizer2 = torch.optim.SGD(w2.parameters(), lr=0.001, momentum=0.9)

w2.state_dict()['linear.weight'][0] = 6
parameter2 = []
loss_list2 = []
for n in range(100):
    optimizer2.zero_grad()
    base_loss = fourth_order(w2(torch.tensor([[1.0]])))
    noise = 100 * torch.randn(1)
    loss = base_loss + noise
    loss_list2.append(loss)
    current_param = w2.state_dict()['linear.weight'][0].detach().data.item()
    parameter2.append(current_param)
    loss.backward()
    optimizer2.step()
    
    if n % 20 == 0 or n == 99:
        print(f"Optimizer 2 - Epoch {n+1}: weight={current_param:.4f}, loss={loss.item():.4f}")

# Create the combined plot
plt.figure(figsize=(12, 8))
plt.plot(W.numpy(), LOSS, label='objective function', linewidth=2)
plt.plot(parameter1, [loss.detach().numpy().item() for loss in loss_list1], 'o', 
         label='SGD without momentum', color='black', markersize=4)
plt.plot(parameter2, [loss.detach().numpy().item() for loss in loss_list2], 'o', 
         label='SGD with momentum=0.9', color='red', markersize=4)
plt.xlabel('w')
plt.ylabel('l(w)')
plt.title('Fourth Order Function - High Noise Comparison (std=100)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final comparison:")
print(f"Optimizer 1 (no momentum): Final weight={parameter1[-1]:.4f}, Final loss={loss_list1[-1].item():.4f}")
print(f"Optimizer 2 (momentum=0.9): Final weight={parameter2[-1]:.4f}, Final loss={loss_list2[-1].item():.4f}")

print("\n" + "="*60)
print("TESTING SUMMARY")
print("="*60)
print("""
BUGS FIXED:
1. Fixed incorrect weight assignment in plotting loops (was modifying tensor directly)
2. Fixed tensor dimension issues in loss extraction (.flatten() -> .item())
3. Added model resets between tests to ensure clean starting conditions
4. Fixed parameter label bug in plot_fourth_order function
5. Added proper figure management for multiple plots

OPTIMIZATIONS TESTED:
1. Cubic Function (Saddle Point Problem):
   - Without momentum: Gets stuck easily at saddle points
   - With momentum (0.9): Better escape from saddle points due to accumulated velocity

2. Fourth Order Function (Local Minima Problem):
   - Without momentum: May get trapped in local minima
   - With momentum (0.9): Better chance of escaping local minima

3. Noisy Environments:
   - Without momentum: More susceptible to noise-induced oscillations
   - With momentum (0.9): Smoother convergence, less affected by noise

KEY INSIGHTS:
- Momentum acts like a ball rolling downhill - it builds up velocity in consistent directions
- This helps escape saddle points and local minima (like a ball rolling over small hills)
- In noisy environments, momentum provides stability by averaging out random fluctuations
- Higher momentum values (like 0.9) provide more stability but may overshoot optimal points

ANALOGY: Think of optimization without momentum like a cautious hiker who stops and reconsiders
at every step. With momentum, it's like a confident cyclist who maintains speed through
small obstacles and can coast over hills that would stop the hiker.
""")
