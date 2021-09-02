# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:37:23 2021

@author: zolta
"""


import model as mo
import dataprocess as dp
import numpy as npy


verbose=1
epochs = 10
batch_size = 32
print("-----------------------------------")


# def testModel(inModel, numRepeats, epochs = 10, batch_size = 32):
#     if(inModel = "")
trainX, trainY, testX, testY = dp.loadHARSet()
print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
scores = []
testModel = mo.testCNN(trainX, trainY, testX, testY)
print(testModel.summary())
for i in range(5):
    testModel = mo.testCNN(trainX, trainY, testX, testY)
    testModel.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, score = testModel.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    
    score = score * 100.0
    print('>#%d: %.3f' % (i+1, score))
    scores.append(score)

trainX, trainY, testX, testY = dp.loadWISDM(mode="at", n_steps=200, step=20, trainSplit=0.2)
print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
testModel = mo.testCNN(trainX, trainY, testX, testY)
print(testModel.summary())
for i in range(5):
    testModel = mo.testCNN(trainX, trainY, testX, testY)
    testModel.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, score = testModel.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    
    score = score * 100.0
    print('>#%d: %.3f' % (i+1, score))
    scores.append(score)


print(scores)
m, s = npy.mean(scores), npy.std(scores)
print('Accuracy: %.3f%% (+/-%.3f)' % (m,s))
modelType = "CNN_test"
dataSet = "UCI HAR"
menu = ''


# while(menu != 'x' or menu != 'X'):
#     print("Current Model selected: "+ modelType)
#     print("Current Dataset slected: " + dataSet)
#     print("Please choose a menu option (x to quit):\n 1 Select Dataset\n 2 Select Model\n 3 test model\n 4 train and save model")
    
#     menu = input("> ")
#     if(menu == '1'):
#         dp.checkAllSets()
#         menu = input("input the number for the model you wish to use\n> ")
#         if(menu == '1'):
#             modelType = "RNN"
#         elif(menu == '2'):
#             modelType = "CNN"
#         elif(menu == '3'):
#             modelType = "CNN_test"
#     elif(menu == '2'):
#         dp.checkAllSets()
#         menu = input("input the number for the database you wish to use\n> ")
#         if(menu == '1'):
#             dataSet = "WISDM_ar"
#         elif(menu == '2'):
#             dataSet = "WISDM_tr"
#         elif(menu == '3'):
#             dataSet = "UCI HAR"
#     elif(menu == '3'):
#         c=1