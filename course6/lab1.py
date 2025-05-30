# concrete crack detection

from PIL import Image
from matplotlib.pyplot import imshow
import pandas
import matplotlib.pylab as plt
import os
import glob
#import skillsnetwork

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

#await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", path = "/resources/data", overwrite=True)

import requests
import zipfile
import os
from pathlib import Path

def prepare_data(url, path="resources/data", overwrite=True):
    """
    Download and extract a zip file to a specified path.
    Similar to skillsnetwork.prepare() functionality.
    """
    # Create the directory if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists and handle overwrite
    if os.path.exists(path) and os.listdir(path) and not overwrite:
        print(f"Data already exists at {path} and overwrite=False")
        return
    
    # Download the file
    print("Downloading data...")
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for bad status codes
    
    # Save to a temporary zip file
    zip_path = os.path.join(path, "temp_download.zip")
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    
    # Clean up the zip file
    os.remove(zip_path)
    print(f"Data prepared successfully at {path}")

# Usage - equivalent to your original async call
#prepare_data(
#    "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip",
#    path="resources/data",
#    overwrite=True
#)

# The path to all the images are stored in the variable directory.
directory="resources/data"

negative='Negative'

negative_file_path=os.path.join(directory,negative)
print(negative_file_path)

print(os.listdir(negative_file_path)[0:3])

print([os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path)][0:3])

print("test.jpg".endswith(".jpg"))
print("test.mpg".endswith(".jpg"))

negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
print(negative_files[0:3])

# now load positives

positive = "Positive"
positive_file_path = os.path.join(directory, positive)
print(positive_file_path)
print(os.listdir(positive_file_path)[0:3])
print([os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path)][0:3])
positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
print(positive_files[0:3])

# Open the first negative image
image1 = Image.open(negative_files[0])
# You can view the image directly
# image1

# Open the first positive image
image2 = Image.open(positive_files[0])
# You can view the image directly
# image2
# You can also display basic image information
print(f"Negative image size: {image1.size}")
print(f"Negative image mode: {image1.mode}")
print(f"Positive image size: {image2.size}")
print(f"Positive image mode: {image2.mode}")

plt.imshow(image1)
plt.title("1st Image With No Cracks")
plt.show()

image2 = Image.open(negative_files[1])
plt.imshow(image2)
plt.title("2nd Image With No Cracks")
plt.show()

# Question 2 - Plot the first three images for the dataset with cracks. Don't forget. You will be asked in the quiz, so remember the image.

import matplotlib.pyplot as plt

# Create a figure with 1 row and 3 columns for the three images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the first three positive (crack) images
for i in range(3):
    # Open each image
    img = Image.open(positive_files[i])

    # Display the image in the corresponding subplot
    axes[i].imshow(img)
    axes[i].set_title(f'Crack Image {i+1}')
    axes[i].axis('off')  # Remove axis ticks and labels

# Adjust spacing between subplots
plt.tight_layout()
plt.show()

# Print the filenames for reference
print("Displayed crack images:")
for i in range(3):
    print(f"Image {i+1}: {positive_files[i]}")


# Examine Files

directory="resources/data"
negative='Negative'
negative_file_path=os.path.join(directory,negative)
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
negative_files[0:3]

positive="Positive"
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
positive_files[0:3]


