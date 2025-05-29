import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("UNDERSTANDING MULTIPLE INPUT AND OUTPUT CHANNELS IN CNNs")
print("="*70)

# ========================
# 1. WHAT ARE CHANNELS?
# ========================
print("\n1. WHAT ARE CHANNELS?")
print("-" * 25)

print("""
CHANNELS are like different "views" or "perspectives" of the same spatial data.

ANALOGY: Think of channels like different color filters on a camera:
- Red channel: Shows red intensity at each pixel
- Green channel: Shows green intensity at each pixel  
- Blue channel: Shows blue intensity at each pixel
- All together: Make a full color image

In CNNs:
- Each channel is a 2D feature map of the same spatial size
- Multiple channels = multiple different feature detectors
""")

# ========================
# 2. INPUT CHANNELS EXAMPLES
# ========================
print("\n2. INPUT CHANNELS - EXAMPLES")
print("-" * 35)

# Example 1: RGB Image (3 input channels)
print("EXAMPLE 1: RGB Image")
rgb_image = torch.randn(3, 32, 32)  # 3 channels, 32x32 pixels
print(f"RGB image shape: {rgb_image.shape}")
print(f"  - Channel 0 (Red): {rgb_image[0].shape}")
print(f"  - Channel 1 (Green): {rgb_image[1].shape}")
print(f"  - Channel 2 (Blue): {rgb_image[2].shape}")

# Example 2: Grayscale Image (1 input channel)
print(f"\nEXAMPLE 2: Grayscale Image (like MNIST)")
gray_image = torch.randn(1, 28, 28)  # 1 channel, 28x28 pixels  
print(f"Grayscale image shape: {gray_image.shape}")
print(f"  - Only 1 channel (intensity): {gray_image[0].shape}")

# Example 3: Feature maps from previous layer (16 input channels)
print(f"\nEXAMPLE 3: Feature maps from previous conv layer")
feature_maps = torch.randn(16, 14, 14)  # 16 channels, 14x14 spatial
print(f"Feature maps shape: {feature_maps.shape}")
print(f"  - 16 different feature detectors")
print(f"  - Each produces a 14x14 activation map")

# ========================
# 3. HOW CONVOLUTION WORKS WITH MULTIPLE INPUT CHANNELS
# ========================
print(f"\n3. HOW CONVOLUTION WORKS WITH MULTIPLE INPUT CHANNELS")
print("-" * 55)

def demonstrate_multichannel_conv():
    """Show how conv works with multiple input channels"""
    
    print("Creating a conv layer: Conv2d(3, 1, kernel_size=3)")
    print("This means:")
    print("  - 3 input channels (e.g., RGB)")
    print("  - 1 output channel (1 filter)")
    print("  - 3x3 kernel size")
    
    conv_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, bias=False)
    
    print(f"\nFilter weights shape: {conv_layer.weight.shape}")
    print("Shape breakdown: [out_channels, in_channels, height, width]")
    print(f"  - {conv_layer.weight.shape[0]} output filter(s)")
    print(f"  - {conv_layer.weight.shape[1]} input channels per filter")
    print(f"  - {conv_layer.weight.shape[2]}x{conv_layer.weight.shape[3]} kernel size")
    
    # Show the actual filter weights
    print(f"\nThe single filter has 3 separate kernels (one for each input channel):")
    for i in range(3):
        print(f"  Kernel for input channel {i}:")
        print(f"    {conv_layer.weight[0, i].detach().numpy()}")
    
    # Apply convolution
    input_tensor = torch.randn(1, 3, 5, 5)  # batch=1, channels=3, 5x5 spatial
    output = conv_layer(input_tensor)
    
    print(f"\nConvolution process:")
    print(f"  Input shape: {input_tensor.shape} [batch, channels, height, width]")
    print(f"  Output shape: {output.shape}")
    print(f"  The filter processes ALL 3 input channels simultaneously")
    print(f"  Result: Single output channel (sum of convolutions from all input channels)")

demonstrate_multichannel_conv()

# ========================
# 4. MULTIPLE OUTPUT CHANNELS
# ========================
print(f"\n4. MULTIPLE OUTPUT CHANNELS")
print("-" * 30)

def demonstrate_multiple_output_channels():
    """Show how multiple output channels work"""
    
    print("Creating a conv layer: Conv2d(3, 16, kernel_size=3)")
    print("This means:")
    print("  - 3 input channels")
    print("  - 16 output channels (16 different filters)")
    print("  - Each filter produces its own output channel")
    
    conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, bias=False)
    
    print(f"\nFilter weights shape: {conv_layer.weight.shape}")
    print("Shape breakdown: [out_channels, in_channels, height, width]")
    print(f"  - {conv_layer.weight.shape[0]} different filters")
    print(f"  - Each filter has {conv_layer.weight.shape[1]} kernels (one per input channel)")
    print(f"  - Each kernel is {conv_layer.weight.shape[2]}x{conv_layer.weight.shape[3]}")
    
    print(f"\nTotal parameters: {conv_layer.weight.numel():,}")
    print(f"Calculation: 16 filters × 3 input channels × 3×3 kernel = {16*3*3*3:,}")
    
    # Apply convolution
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch=1, RGB image 32x32
    output = conv_layer(input_tensor)
    
    print(f"\nConvolution process:")
    print(f"  Input: {input_tensor.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Each of the 16 filters produces its own 30x30 feature map")
    print(f"  (32-3+1=30, assuming no padding)")

demonstrate_multiple_output_channels()

# ========================
# 5. VISUAL DEMONSTRATION
# ========================
print(f"\n5. VISUAL DEMONSTRATION")
print("-" * 25)

def visualize_channels():
    """Create a visual demonstration of channels"""
    
    # Create a simple synthetic RGB image
    rgb = torch.zeros(3, 10, 10)
    
    # Red channel: vertical lines
    rgb[0, :, [2, 7]] = 1.0
    
    # Green channel: horizontal lines  
    rgb[1, [3, 6], :] = 1.0
    
    # Blue channel: diagonal
    for i in range(10):
        if i < 10:
            rgb[2, i, i] = 1.0
    
    # Create conv layer with different filters
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
    
    # Manually set some interpretable filters
    with torch.no_grad():
        # Filter 0: Vertical edge detector (mainly uses red channel)
        conv.weight[0, 0] = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        conv.weight[0, 1] = torch.zeros(3, 3)
        conv.weight[0, 2] = torch.zeros(3, 3)
        
        # Filter 1: Horizontal edge detector (mainly uses green channel)
        conv.weight[1, 0] = torch.zeros(3, 3)
        conv.weight[1, 1] = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        conv.weight[1, 2] = torch.zeros(3, 3)
        
        # Filter 2: Diagonal detector (mainly uses blue channel)
        conv.weight[2, 0] = torch.zeros(3, 3)
        conv.weight[2, 1] = torch.zeros(3, 3)
        conv.weight[2, 2] = torch.tensor([[1, 0, -1], [0, 1, 0], [-1, 0, 1]], dtype=torch.float32)
        
        # Filter 3: Combination detector (uses all channels)
        conv.weight[3, 0] = torch.ones(3, 3) * 0.33
        conv.weight[3, 1] = torch.ones(3, 3) * 0.33
        conv.weight[3, 2] = torch.ones(3, 3) * 0.33
    
    # Apply convolution
    input_batch = rgb.unsqueeze(0)  # Add batch dimension
    output = conv(input_batch)
    output = output.squeeze(0)  # Remove batch dimension
    
    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot input channels
    channels = ['Red', 'Green', 'Blue']
    for i in range(3):
        im = axes[0, i].imshow(rgb[i].numpy(), cmap='gray')
        axes[0, i].set_title(f'Input Channel {i} ({channels[i]})')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i])
    
    # Plot combined RGB
    rgb_combined = rgb.permute(1, 2, 0).numpy()
    axes[0, 3].imshow(rgb_combined)
    axes[0, 3].set_title('Combined RGB')
    axes[0, 3].axis('off')
    
    # Plot output channels
    filter_names = ['Vertical Edge', 'Horizontal Edge', 'Diagonal', 'Combined']
    for i in range(4):
        im = axes[1, i].imshow(output[i].detach().numpy(), cmap='viridis')
        axes[1, i].set_title(f'Output Channel {i}\n({filter_names[i]})')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.show()
    
    return rgb, output

print("Creating visual demonstration...")
input_channels, output_channels = visualize_channels()

# ========================
# 6. CHANNEL PROGRESSION IN A TYPICAL CNN
# ========================
print(f"\n6. CHANNEL PROGRESSION IN A TYPICAL CNN")
print("-" * 45)

def show_channel_progression():
    """Show how channels evolve through a CNN"""
    
    print("Typical channel progression in a CNN for image classification:")
    print()
    
    # Define a typical CNN architecture
    layers = [
        ("Input", "RGB Image", (3, 224, 224)),
        ("Conv1", "Basic features", (64, 224, 224)),
        ("Pool1", "Downsampled", (64, 112, 112)),
        ("Conv2", "Complex features", (128, 112, 112)),
        ("Pool2", "Downsampled", (128, 56, 56)),
        ("Conv3", "High-level features", (256, 56, 56)),
        ("Pool3", "Downsampled", (256, 28, 28)),
        ("Conv4", "Abstract features", (512, 28, 28)),
        ("Pool4", "Downsampled", (512, 14, 14)),
        ("Global Avg Pool", "Spatial summary", (512, 1, 1)),
        ("FC", "Classification", (1000,))
    ]
    
    print(f"{'Layer':<15} {'Description':<20} {'Shape':<15} {'Parameters'}")
    print("-" * 70)
    
    for i, (layer, desc, shape) in enumerate(layers):
        if len(shape) == 3:
            shape_str = f"{shape[0]}×{shape[1]}×{shape[2]}"
        else:
            shape_str = f"{shape[0]}"
            
        # Calculate parameters for conv layers (simplified)
        if layer.startswith("Conv"):
            prev_channels = layers[i-1][2][0] if i > 0 else 3
            curr_channels = shape[0]
            params = f"{prev_channels*curr_channels*9:,}"  # Assuming 3x3 kernels
        else:
            params = "-"
            
        print(f"{layer:<15} {desc:<20} {shape_str:<15} {params}")
    
    print(f"\nKey observations:")
    print(f"1. Channels typically INCREASE as we go deeper (3→64→128→256→512)")
    print(f"2. Spatial dimensions DECREASE due to pooling (224→112→56→28→14)")
    print(f"3. More channels = more specialized feature detectors")
    print(f"4. Deeper layers detect more abstract/complex features")

show_channel_progression()

# ========================
# 7. PRACTICAL IMPLICATIONS
# ========================
print(f"\n7. PRACTICAL IMPLICATIONS")
print("-" * 30)

print("""
UNDERSTANDING CHANNELS HELPS WITH:

1. ARCHITECTURE DESIGN:
   - How many filters do I need in each layer?
   - Balance between model capacity and computational cost
   - Common patterns: 32→64→128→256 or 16→32→64→128

2. PARAMETER COUNTING:
   Conv2d(in_channels, out_channels, kernel_size) parameters:
   = out_channels × in_channels × kernel_height × kernel_width + out_channels (bias)
   
   Example: Conv2d(3, 64, 3) = 64 × 3 × 3 × 3 + 64 = 1,792 parameters

3. MEMORY USAGE:
   More channels = more memory for storing activations
   Feature map memory = batch_size × channels × height × width × 4 bytes (float32)

4. COMPUTATIONAL COST:
   More input/output channels = more multiply-accumulate operations
   FLOPs ≈ output_height × output_width × kernel_size² × in_channels × out_channels

5. FEATURE LEARNING:
   - First layers: Simple features (edges, colors)
   - Middle layers: Shapes, textures, patterns  
   - Deep layers: Object parts, abstract concepts
   - Each channel specializes in different aspects

6. DEBUGGING:
   - Visualize individual channels to understand what network learned
   - Check if all channels are being used (not dead)
   - Monitor channel statistics during training
""")

# ========================
# 8. HANDS-ON EXAMPLES
# ========================
print(f"\n8. HANDS-ON EXAMPLES")
print("-" * 22)

print("Let's trace through our MNIST CNN:")
print()

# Recreate the MNIST CNN structure
print("MNIST CNN Channel Flow:")
print("Input:  1 channel  (28×28) - Grayscale digit")
print("Conv1:  1→16 channels (28×28) - 16 basic edge/curve detectors")  
print("Pool1:  16 channels (14×14) - Downsampled")
print("Conv2:  16→32 channels (14×14) - 32 complex pattern detectors")
print("Pool2:  32 channels (7×7) - Downsampled")
print("FC:     32×7×7=1568 → 128 → 10")

print(f"\nParameter calculation:")
conv1_params = 1 * 16 * 3 * 3 + 16  # +bias
conv2_params = 16 * 32 * 3 * 3 + 32  # +bias
fc1_params = 1568 * 128 + 128
fc2_params = 128 * 10 + 10

print(f"Conv1: {conv1_params:,} parameters (1×16×3×3 + 16 bias)")
print(f"Conv2: {conv2_params:,} parameters (16×32×3×3 + 32 bias)")
print(f"FC1:   {fc1_params:,} parameters (1568×128 + 128 bias)")
print(f"FC2:   {fc2_params:,} parameters (128×10 + 10 bias)")
print(f"Total: {conv1_params + conv2_params + fc1_params + fc2_params:,} parameters")

print(f"\n" + "="*70)
print("SUMMARY: MULTIPLE CHANNELS ARE LIKE MULTIPLE SPECIALISTS")
print("="*70)
print("""
Think of channels like having multiple specialists examining the same data:

INPUT CHANNELS = Multiple perspectives of the same scene
- RGB image: Red specialist, Green specialist, Blue specialist
- Previous layer: Edge specialist, Texture specialist, Shape specialist...

OUTPUT CHANNELS = Multiple specialized detectors
- Each filter creates its own specialty detection map
- More filters = more specialized feature detectors
- Network learns what specializations are most useful

The magic: Each filter looks at ALL input channels simultaneously,
then creates ONE output channel representing its specialized detection.

This is why CNNs are so powerful - they automatically learn
the right specialists for the task!
""")
