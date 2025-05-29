import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def calculate_dataset_stats(dataset_loader, dataset_name="Unknown"):
    """
    Calculate mean and standard deviation for a dataset.
    This is like finding the "average pixel value" and "typical variation"
    across all images in your dataset.
    
    Think of it as:
    - Mean: What's the typical brightness level?
    - Std: How much do pixel values typically vary from that average?
    """
    print(f"\nCalculating statistics for {dataset_name} dataset...")
    
    # Initialize accumulators
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    total_pixels = 0
    
    # For tracking progress
    batch_count = 0
    
    for batch_idx, (data, _) in enumerate(dataset_loader):
        # data shape: [batch_size, channels, height, width]
        batch_samples = data.size(0)  # Number of images in this batch
        
        # Flatten each image: [batch, channels, height, width] -> [batch, channels, height*width]
        data = data.view(batch_samples, data.size(1), -1)
        
        # Sum across batch and spatial dimensions, keep channel dimension
        # This gives us the sum for each channel across all pixels in this batch
        pixel_sum += data.sum(dim=[0, 2])  # Sum over batch and spatial dims
        pixel_squared_sum += (data ** 2).sum(dim=[0, 2])
        
        # Count total pixels processed (for each channel)
        total_pixels += batch_samples * data.size(2)  # batch_size * (height * width)
        
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"Processed {batch_count} batches...")
    
    # Calculate mean and std for each channel
    mean = pixel_sum / total_pixels
    # Std formula: sqrt(E[X²] - E[X]²)
    std = torch.sqrt(pixel_squared_sum / total_pixels - mean ** 2)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Total batches processed: {batch_count}")
    print(f"Total pixels per channel: {total_pixels:,}")
    
    # Convert to lists for easy copying
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    if len(mean_list) == 1:
        # Grayscale image
        print(f"Mean: {mean_list[0]:.4f}")
        print(f"Std:  {std_list[0]:.4f}")
        print(f"\nFor transforms.Normalize: ({mean_list[0]:.4f},), ({std_list[0]:.4f},)")
    else:
        # RGB image
        print(f"Mean per channel (R, G, B): {[f'{m:.4f}' for m in mean_list]}")
        print(f"Std per channel (R, G, B):  {[f'{s:.4f}' for s in std_list]}")
        print(f"\nFor transforms.Normalize: {tuple(round(m, 4) for m in mean_list)}, {tuple(round(s, 4) for s in std_list)}")
    
    return mean_list, std_list

# ========================
# Example 1: MNIST (to verify our method works)
# ========================
print("="*60)
print("EXAMPLE 1: Calculating MNIST statistics")
print("="*60)

# Load MNIST WITHOUT normalization first (just convert to tensor)
transform_no_norm = transforms.Compose([transforms.ToTensor()])

mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_no_norm)
mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=False)

mnist_mean, mnist_std = calculate_dataset_stats(mnist_loader, "MNIST")

print(f"\nComparison with known MNIST values:")
print(f"Our calculated:  mean={mnist_mean[0]:.4f}, std={mnist_std[0]:.4f}")
print(f"Known values:    mean=0.1307, std=0.3081")
print(f"Difference:      mean={abs(mnist_mean[0] - 0.1307):.6f}, std={abs(mnist_std[0] - 0.3081):.6f}")

# ========================
# Example 2: CIFAR-10 (RGB images)
# ========================
print("\n" + "="*60)
print("EXAMPLE 2: Calculating CIFAR-10 statistics")
print("="*60)

# Load CIFAR-10 WITHOUT normalization
cifar10_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_no_norm)
cifar10_loader = DataLoader(cifar10_dataset, batch_size=64, shuffle=False)

cifar10_mean, cifar10_std = calculate_dataset_stats(cifar10_loader, "CIFAR-10")

print(f"\nComparison with commonly used CIFAR-10 values:")
print(f"Our calculated:  mean={[f'{m:.4f}' for m in cifar10_mean]}, std={[f'{s:.4f}' for s in cifar10_std]}")
print(f"Common values:   mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]")

# ========================
# Example 3: Custom dataset function
# ========================
def analyze_custom_dataset(dataset_path_or_dataset, batch_size=64):
    """
    Generic function to analyze any dataset.
    
    Usage examples:
    1. For torchvision datasets: analyze_custom_dataset(your_dataset)
    2. For custom datasets: analyze_custom_dataset(your_custom_dataset)
    """
    print("\n" + "="*60)
    print("ANALYZING CUSTOM DATASET")
    print("="*60)
    
    # Create dataloader
    if hasattr(dataset_path_or_dataset, '__len__'):
        # It's already a dataset
        dataset = dataset_path_or_dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        print("Error: Please provide a PyTorch dataset object")
        return None, None
    
    return calculate_dataset_stats(loader, "Custom Dataset")

# ========================
# WHY NORMALIZATION MATTERS
# ========================
print("\n" + "="*60)
print("WHY NORMALIZATION MATTERS")
print("="*60)

def demonstrate_normalization_effect():
    """Show the effect of normalization on pixel values"""
    
    # Get a sample batch from MNIST
    sample_data, _ = next(iter(mnist_loader))
    
    print("Raw pixel values (first image, first 20 pixels):")
    raw_pixels = sample_data[0, 0, 0, :20]  # First image, first row, first 20 pixels
    print([f"{p:.3f}" for p in raw_pixels])
    print(f"Raw data range: {sample_data.min():.3f} to {sample_data.max():.3f}")
    print(f"Raw data mean: {sample_data.mean():.4f}, std: {sample_data.std():.4f}")
    
    # Apply normalization
    normalize = transforms.Normalize((mnist_mean[0],), (mnist_std[0],))
    normalized_data = normalize(sample_data)
    
    print(f"\nAfter normalization:")
    norm_pixels = normalized_data[0, 0, 0, :20]
    print([f"{p:.3f}" for p in norm_pixels])
    print(f"Normalized range: {normalized_data.min():.3f} to {normalized_data.max():.3f}")
    print(f"Normalized mean: {normalized_data.mean():.4f}, std: {normalized_data.std():.4f}")
    
    print(f"\nNormalization formula applied: (pixel - {mnist_mean[0]:.4f}) / {mnist_std[0]:.4f}")
    print("This centers the data around 0 and scales to unit variance")
    print("Benefits:")
    print("- Faster training convergence")
    print("- More stable gradients")
    print("- Better optimization performance")
    print("- Prevents certain channels/features from dominating")

demonstrate_normalization_effect()

# ========================
# PRACTICAL TIPS
# ========================
print("\n" + "="*60)
print("PRACTICAL TIPS")
print("="*60)

print("""
1. WHEN TO CALCULATE STATS:
   - Always calculate on TRAINING set only (not test set)
   - Calculate before any other transformations except ToTensor()
   - Use the same stats for train, validation, and test sets

2. FOR DIFFERENT IMAGE TYPES:
   - Grayscale: Single mean/std value
   - RGB: Three values (one per channel)
   - Different datasets need different stats!

3. COMPUTATIONAL EFFICIENCY:
   - Use smaller batch size if memory is limited
   - Can sample subset of data for approximation (use random subset)
   - Cache the results - no need to recalculate every time

4. CODE TEMPLATE FOR YOUR DATASETS:
   ```python
   # Replace with your dataset
   your_dataset = YourCustomDataset(transform=transforms.ToTensor())
   your_loader = DataLoader(your_dataset, batch_size=64, shuffle=False)
   mean, std = calculate_dataset_stats(your_loader, "Your Dataset Name")
   
   # Then use in your transforms:
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ])
   ```

5. VERIFICATION:
   - After applying normalization, your data should have ~0 mean and ~1 std
   - Small deviations are normal due to batch effects
""")

print("\nRemember: These statistics are like the 'fingerprint' of your dataset.")
print("Different datasets have different characteristics, so always calculate fresh!")

