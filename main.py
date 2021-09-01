# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:37:23 2021

@author: zolta
"""

import tensorflow as tf
import model as mo
import dataprocess as dp
import numpy as npy


trainX, trainY, testX, testY = dp.loadHARSet()
verbose=0
epochs = 10
batch_size = 32
print("-----------------------------------")
print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)


scores = []
for i in range(10):
    testModel = mo.testCNN(trainX, trainY, testX, testY)
    testModel.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, score = testModel.evaluate(testX, testY, batch_size=batch_size, verbose=0)
    
    score = score * 100.0
    print('>#%d: %.3f' % (i+1, score))
    scores.append(score)
    
print(scores)
m, s = npy.mean(scores), npy.std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m,s))