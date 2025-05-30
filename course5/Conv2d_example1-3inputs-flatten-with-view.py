import torch

# Let's say we have a batch of 2 images for simplicity
x = torch.randn(2, 128, 4, 4)  # 2 images, 128 channels, 4x4 each
print(f"Before flatten: {x.shape}")  # [2, 128, 4, 4]

x = x.view(-1, 128 * 4 * 4)
print(f"After flatten: {x.shape}")   # [2, 2048]

# What happened:
# Image 1: 128 channels × 4×4 = 2048 values → becomes row 1
# Image 2: 128 channels × 4×4 = 2048 values → becomes row 2
