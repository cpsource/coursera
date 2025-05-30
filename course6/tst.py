import os
from PIL import Image
import matplotlib.pyplot as plt

# Check if files actually exist
print("Checking if files exist:")
for i, file_path in enumerate(all_files[0:4]):
    exists = os.path.exists(file_path)
    print(f"File {i}: {file_path}")
    print(f"  Exists: {exists}")
    if exists:
        print(f"  File size: {os.path.getsize(file_path)} bytes")
    print()

# Try to open each image individually with error handling
print("Trying to open images:")
for i, file_path in enumerate(all_files[0:4]):
    try:
        print(f"Opening image {i}: {file_path}")
        img = Image.open(file_path)
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")
        
        # Try to display it
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"y={Y[i].item()}")
        plt.axis('off')
        plt.show()
        
    except FileNotFoundError:
        print(f"  ERROR: File not found!")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
