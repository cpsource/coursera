# Pre-Trained Models

import skillsnetwork

#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

import requests
import zipfile
import os
from pathlib import Path

# Add these lines at the very beginning of lab4.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage to avoid GPU issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Reduce TensorFlow logging

# Pre-Trained Models
import skillsnetwork

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

# download data

# ## get the data
#await skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip", overwrite=True)

# The path to all the images are stored in the variable directory.
use_directory="concrete_data_week3"

def prepare_data(url, path=use_directory, overwrite=True):
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

if False:
    # Usage - equivalent to your original async call
    prepare_data(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip",
        path=use_directory,
        overwrite=True
    )

# Global Constants

num_classes = 2
image_resize = 224
batch_size_training = 100
batch_size_validation = 100

# Construct ImageDataGenerator Instances

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

# Next, we will use the flow_from_directory method to get the training images as follows:

print("... running train_generator")
train_generator = data_generator.flow_from_directory(
    use_directory + "/concrete_data_week3/" + "train",
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')

# Use the flow_from_directory method to get the validation images
#  and assign the result to validation_generator.

print("... running validation_generator")
validation_generator = data_generator.flow_from_directory(
    use_directory + "/concrete_data_week3/" + "valid",
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')

print("...complete")

# Build, Compile and Fit Model

model = Sequential()

model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))

model.add(Dense(num_classes, activation='softmax'))

print(model.layers)

# You can access the ResNet50 layers by running the following:

print(model.layers[0].layers)

# Since the ResNet50 model has already been trained, then we want to tell our model not to bother with training the ResNet part, but to train only our dense output layer. To do that, we run the following.

model.layers[0].trainable = False

# And now using the summary attribute of the model, we can see how many parameters we will need to optimize in order to train the output layer.

print(model.summary())

# Next we compile our model using the adam optimizer.

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2

# Finally, we are ready to start training our model. Unlike a conventional deep learning training were data is not streamed from a directory, with an ImageDataGenerator where data is augmented in batches, we use the fit_generator method.

# Replace this section in your code (around line 136):

# OLD - DEPRECATED:
# fit_history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=steps_per_epoch_training,
#     epochs=num_epochs,
#     validation_data=validation_generator,
#     validation_steps=steps_per_epoch_validation,
#     verbose=1,
# )

if True:
    # You also need to compile the model before training!
    # Add this BEFORE the fit() call:
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# NEW - MODERN APPROACH:
fit_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)

# save
model.save('classifier_resnet_model.h5')

