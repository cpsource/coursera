# Data preparation with pytorch

from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

directory="resources/data"
negative='Negative'
negative_file_path=os.path.join(directory,negative)
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
print(negative_files[0:3])

positive="Positive"
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
print(positive_files[0:3])

#Question 1
#Find the combined length of the list positive_files and negative_files using the function len . Then assign it to the variable number_of_samples

number_of_samples = len(positive_files) + len(negative_files)
print(f"Total number of samples: {number_of_samples}")

# Assign Lables to Images

Y=torch.zeros([number_of_samples])
Y=Y.type(torch.LongTensor)
print(Y.type())
Y[::2]=1
Y[1::2]=0

# Question 2
# Create a list all_files such that the even indexes contain the path to images with positive or cracked samples and the odd element contain the negative images or images with out cracks. Then use the following code to print out the first four samples.

print("Question 2")

all_files = []
for i in range(max(len(positive_files), len(negative_files))):
    if i < len(positive_files):
        all_files.append(positive_files[i])  # Even indexes (0, 2, 4, ...) - positive/cracked
    if i < len(negative_files):
        all_files.append(negative_files[i])  # Odd indexes (1, 3, 5, ...) - negative/no cracks

for y,file in zip(Y, all_files[0:4]):
    plt.imshow(Image.open(file))
    plt.title("y="+str(y.item()))
    plt.show()

print(f"len all_files = {len(all_files)}")
print(f"len Y = {len(Y)}")

for y,file in zip(Y, all_files[0:4]):
    plt.imshow(Image.open(file))
    plt.title("y="+str(y.item()))
    plt.show()

# Training and Validation Split

train=False

if train:
    all_files=all_files[0:20000]
    Y=Y[0:20000]
else:
    all_files=all_files[20000:]
    Y=Y[20000:]

print(f"len all_files = {len(all_files)}")
print(f"len Y = {len(Y)}")

#Question 3

#Modify the above lines of code such that if the variable train is set to True the first 20000 samples of all_files are use in training. If train is set to False the remaining samples are used for validation. In both cases reassign the values to the variable all_files, then use the following lines of code to print out the first four validation sample images.

#Just a note the images printed out in question two are the first four training samples.

print("start q3")
print(f"len all_files = {len(all_files)}")
print(f"len Y = {len(Y)}")

# Print the first four samples (will be validation samples since train=False)
for y, file in zip(Y[0:4], all_files[0:4]):
    
    print(f"y = {y.item()}, file = {file}")
    
    plt.imshow(Image.open(file))
    plt.title("y=" + str(y.item()))
    plt.show()
    
print("done q3")

# Create a Dataset Class

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="resources/data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()

        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0

        if train:
            self.Y=self.Y[0:20000]
            self.len=len(self.all_files)
        else:
            self.Y=self.Y[20000:]
            self.len=len(self.all_files)

    # Get the length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

print("First 4")

dataset = Dataset()
samples = [9,99]

for sample  in samples:
    plt.imshow(dataset[sample][0])
    plt.xlabel("y="+str(dataset[sample][1].item()))
    plt.title("training data, sample {}".format(int(sample)))
    plt.show()

dataset = Dataset(train=False)

print("Five ")

samples = [15, 102]

for sample  in samples:
    plt.imshow(dataset[sample][0])
    plt.xlabel("y="+str(dataset[sample][1].item()))
    plt.title("training data, sample {}".format(int(sample)))
    plt.show()

#dd`import sys
#sys.exit(0)

