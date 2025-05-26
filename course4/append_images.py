from PIL import Image

# Load both images
img1 = Image.open("image1.png")
img2 = Image.open("image2.png")

# Match width/height if needed (optional resize for alignment)
# img2 = img2.resize(img1.size)

# Horizontally append (side by side)
combined_horizontal = Image.new('RGB', (img1.width + img2.width, max(img1.height, img2.height)))
combined_horizontal.paste(img1, (0, 0))
combined_horizontal.paste(img2, (img1.width, 0))
combined_horizontal.save("combined_horizontal.png")

# Vertically append (top to bottom)
combined_vertical = Image.new('RGB', (max(img1.width, img2.width), img1.height + img2.height))
combined_vertical.paste(img1, (0, 0))
combined_vertical.paste(img2, (0, img1.height))
combined_vertical.save("combined_vertical.png")

