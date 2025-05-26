import torch
from torch.nn import Linear
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(1)

# Create a linear model with 2 input features and 1 output
model = Linear(in_features=2, out_features=1)

# List model parameters (weights and bias)
for param in model.parameters():
    print(param)
print(model.state_dict())

#
# run 1
#
X = torch.tensor([[1.0,3.0]])
yhat = model(X)
print(yhat)

torch.manual_seed(1) # make yhat the same for run2 and LR

#
# run 2
#
X=torch.tensor([[1.0,1.0],[1.0,2.0],[1.0,3.0]])
yhat = model(X)
print(yhat)

#
# Use a class
#

torch.manual_seed(1)
class LR(nn.Module):
    def __init__(self,input_size,output_size):
        super(LR,self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self,x):
        out=self.linear(x)
        return out

model = LR(2,1)
yhat = model(X)
print(yhat)

