
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

n_cols = concrete_data.shape[1]

model.add(Dense(5,activation='relu',input_shape=(n_cols,)))
model.add(Dense(5,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(predictors, target)

predictions = model.predict(test_data)
