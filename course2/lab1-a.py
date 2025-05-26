import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Input value
x = 0.2675
result = sigmoid(x)

print(f"Sigmoid({x}) = {result:.4f}")
