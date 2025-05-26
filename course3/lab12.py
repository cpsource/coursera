# Lab: Hyperparameter Tuning with Keras Tuner

import sys

# Increase recursion limit to prevent potential issues
sys.setrecursionlimit(100000)

# Step 2: Import necessary libraries
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import os
import warnings

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = ERROR only

from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(x_all, y_all), _ = keras.datasets.mnist.load_data()

# Flatten and normalize the images
x_all = x_all.reshape((x_all.shape[0], -1)).astype("float32") / 255.0

# Split into train+val and test (80/20)
x_temp, x_test, y_temp, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

# Split train+val into train and validation (75/25 of 80% = 60/20 overall)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

# Step 3: Load and preprocess the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')

#Exercise 2: Defining the model with hyperparameters

# Define a model-building function

def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Exercise 3: Configuring the hyperparameter search

# Create a RandomSearch Tuner

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space
tuner.search_space_summary()

# Exercise 4: Running the hyperparameter search

# Run the hyperparameter search
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Display a summary of the results
tuner.results_summary()

# Exercise 5: Analyzing and using the best hyperparameters

# Step 1: Retrieve the best hyperparameters 

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 
print(f""" 

The optimal number of units in the first dense layer is {best_hps.get('units')}. 

The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 

""") 

# Step 2: Build and Train the Model with Best Hyperparameters 
model = tuner.hypermodel.build(best_hps) 
model.fit(x_train, y_train, epochs=10, validation_split=0.2) 

# Evaluate the model on the test set 
test_loss, test_acc = model.evaluate(x_val, y_val) 
print(f'Test accuracy: {test_acc}') 

# Exercise 1: Setting Up Keras Tuner

# Step 2: Import necessary libraries
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

# Step 3: Load and preprocess the MNIST data set
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

# Print the shapes of the training and validation datasets
print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')

# Exercise 2: Defining the model with hyperparameters

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Step 1: Define a model-building function
def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

#Exercise 3: Configuring the hyperparameter search

import keras_tuner as kt

# Step 1: Create a RandomSearch Tuner
tuner = kt.RandomSearch(
    build_model,  # Ensure 'build_model' function is defined from previous code
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space
tuner.search_space_summary()

# Exercise 4: Running the hyperparameter search

# Step 1: Run the hyperparameter search

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

 # Display a summary of the results

tuner.results_summary()

# Exercise 5: Analyzing and using the best hyperparameters

# Step 1: Retrieve the best hyperparameters

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""

The optimal number of units in the first dense layer is {best_hps.get('units')}.

The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.

""")

 # Step 2: Build and train the model with best hyperparameters

model = tuner.hypermodel.build(best_hps)

model.fit(x_train, y_train, epochs=10, validation_split=0.2)

 # Evaluate the model on the validation set

val_loss, val_acc = model.evaluate(x_val, y_val)

print(f'Validation accuracy: {val_acc}')

