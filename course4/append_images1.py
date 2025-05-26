from PIL import Image

# Ask for user input
img1_path = input("Enter the path to the first image (.png): ")
img2_path = input("Enter the path to the second image (.png): ")

# Load images
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# Optional: Resize second image to match first (or vice versa)
# Uncomment if needed to force alignment
# img2 = img2.resize(img1.size)

# Horizontally append
combined_horizontal = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
combined_horizontal.paste(img1, (0, 0))
combined_horizontal.paste(img2, (img1.width, 0))
combined_horizontal.save("combined_horizontal.png")
print("✅ Saved combined_horizontal.png")

# Vertically append
combined_vertical = Image.new('RGB', (max(img1.width, img2.width), img1.height + img2.height))
combined_vertical.paste(img1, (0, 0))
combined_vertical.paste(img2, (0, img1.height))
combined_vertical.save("combined_vertical.png")
print("✅ Saved combined_vertical.png")

