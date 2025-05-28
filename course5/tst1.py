import torch
import torch.nn as nn

# Simple network with one hidden layer
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 3)
        self.classifier = nn.Linear(3, 1)  # Linear classifier
        
    def forward(self, x):
        h = torch.relu(self.hidden(x))  # Non-linear activation!
        output = self.classifier(h)     # Linear transformation
        return output

model = SimpleNet()

# Let's see what happens with some inputs
x1 = torch.tensor([[1.0, 2.0]])
x2 = torch.tensor([[2.0, 4.0]])  # Exactly double x1

y1 = model(x1)
y2 = model(x2)

print(f"x1: {x1}, output: {y1}")
print(f"x2: {x2}, output: {y2}")
print(f"Is y2 = 2 * y1? {torch.allclose(y2, 2 * y1)}")  # Usually False!
