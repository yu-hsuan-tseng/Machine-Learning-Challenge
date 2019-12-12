'''
    machine learning challenge
    day 93
'''
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense

print(keras.__version__)
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))

model.add(layers.Conv2D(25,(3,3)))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(units=100))
model.add(Activation('relu'))

model.add(Dense(units=10))
model.add(Activation('softmax'))
model.summary()

