# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:41:23 2021

@author: zolta
"""
import tensorflow as tf
from tensorflow import keras as ke
from tensorflow.keras import layers


def CNNModel(trainX, trainY, testX, testY):
    model = ke.models.Sequential()
    
    return model


def RNNModel(trainX, trainY, testX, testY):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    
    model = ke.models.Sequential()
    weight = tf.keras.initializers.RandomNormal()
    
    model.add(layers.LSTM(60, input_shape=(n_timesteps, n_features)))
    model.add(layers.LSTM(60))
    model.add(layers.LSTM(60))
    
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def testCNN(trainX, trainY, testX, testY):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    
    model = ke.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
       
    return model