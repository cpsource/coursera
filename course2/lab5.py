# regression models with keras

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import keras

import warnings
warnings.simplefilter('ignore', FutureWarning)

# download and clean the data set
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

print(concrete_data.head())
print(concrete_data.shape)
print(concrete_data.describe())
print(concrete_data.isnull().sum())

# split
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# quick sanity check
print(predictors.head())
print(target.head())

# normalize
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors_norm.head())

# save number of predictors
n_cols = predictors_norm.shape[1] # number of predictors

# Import Keras Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

# build a NN
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# train and test
# build the model
model = regression_model()

# fit the model
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# Practice Exercise1
def regression_model():
    input_colm = predictors_norm.shape[1] # Number of input features
    # create model
    model = Sequential()
    model.add(Input(shape=(input_colm,)))  # Set the number of input features
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Output layer

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Practice Exercise 2

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=2)

