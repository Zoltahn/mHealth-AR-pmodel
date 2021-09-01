# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:41:23 2021

@author: zolta
"""

import tensorflow as tf
from tensorflow import keras as ke
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D



def CNNModel():
    model = ke.models.Sequential()
    
    return model


def RNNModel():
    model = ke.models.Sequential()
    
    return model

def testCNN(trainX, trainY, testX, testY):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   	# fit network
    
       
    return model