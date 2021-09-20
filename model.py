# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:41:23 2021

@author: zolta
"""
import tensorflow as tf
from tensorflow import keras as ke
from tensorflow.keras import layers


def CNNModel(trainX, trainY):
    numSteps, numFeat, numOut = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    
    headOne = ke.Input(shape=(numSteps, numFeat))
    headOneConv = layers.Conv1D(filters=64, kernel_size=3, activation ='relu')(headOne)
    headOneDrop = layers.Dropout(0.5)(headOneConv)
    headOnePool = layers.MaxPooling1D(pool_size=2)(headOneDrop)
    headOneFlat = layers.Flatten()(headOnePool)
    
    headTwo = ke.Input(shape=(numSteps, numFeat))
    headTwoConv = layers.Conv1D(filters=64, kernel_size=5, activation ='relu')(headTwo)
    headTwoDrop = layers.Dropout(0.5)(headTwoConv)
    headTwoPool = layers.MaxPooling1D(pool_size=2)(headTwoDrop)
    headTwoFlat = layers.Flatten()(headTwoPool)
    
    headThree = ke.Input(shape=(numSteps, numFeat))
    headThreeConv = layers.Conv1D(filters=64, kernel_size=7, activation ='relu')(headThree)
    headThreeDrop = layers.Dropout(0.5)(headThreeConv)
    headThreePool = layers.MaxPooling1D(pool_size=2)(headThreeDrop)
    headThreeFlat = layers.Flatten()(headThreePool)
    
    heads = layers.concatenate([headOneFlat, headTwoFlat, headThreeFlat])
    
    denseLayer = layers.Dense(100, activation='relu')(heads)
    output = layers.Dense(numOut, activation='softmax')(denseLayer)
    model = ke.models.Model(inputs=[headOne, headTwo, headThree], outputs = output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model = ke.models.Sequential()
    
    return model


def RNNModel(trainX, trainY):
    numSteps, numFeat, numOut = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    
    model = ke.models.Sequential()
    weight = tf.keras.initializers.RandomNormal()

    model.add(layers.LSTM(60, input_shape=(numSteps, numFeat), return_sequences=True, kernel_initializer=weight))
    model.add(layers.LSTM(60, return_sequences=True))
    model.add(layers.LSTM(60))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(numOut, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def testCNN(trainX, trainY):
    numSteps, numFeat, numOut = trainX.shape[1], trainX.shape[2], trainY.shape[1]

    
    model = ke.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(numSteps,numFeat)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(numOut, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
       
    return model