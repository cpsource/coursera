import torch

a = torch.tensor([4.0,3.5])

print(a.dtype)

print(a.type())

a = torch.tensor([4, 1, 2, 3, 4])
b=a.type(torch.float32)
print(b.dtype)

print(a.size())
print(a.ndimension())

a_col = a.view(-1,1)
print(a.ndimension())

import torch

# 1D tensor
tensor_1d = torch.tensor([1, 2, 3, 4])  # shape: [4]

# Option 1: Unsqueeze to add a batch dimension
tensor_2d = tensor_1d.unsqueeze(0)     # shape: [1, 4]

# Option 2: Use .view or .reshape
tensor_2d_alt = tensor_1d.view(1, -1)  # shape: [1, 4]

print(tensor_2d.shape)       # torch.Size([1, 4])
print(tensor_2d_alt.shape)   # torch.Size([1, 4])

print(tensor_2d)

print(tensor_1d.view(1,-1))
print(tensor_1d.view(-1,4))

