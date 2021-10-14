# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:37:23 2021

@author: zolta
"""


import model as mo
import dataprocess as dp
import numpy as npy
import tensorflow as tf


verbose=1
epochs = 10
batch_size = 32
WISDM_BATCH_SIZE_DEF = 200
WISDM_STEP_SIZE_DEF = 20
WISDM_TRAIN_RATIO_DEF = 0.2

def configWISDM():
    print("Default settings:\n signal batch size -> " + str(WISDM_BATCH_SIZE_DEF) + 
          "\nstep gap between signal batches -> " + str(WISDM_STEP_SIZE_DEF) + 
          "\ntrain/test validation split -> " + str(WISDM_TRAIN_RATIO_DEF))
    cc = input ("use custom settings for dataset batch sizes?\n\n(y/n)> ")
    cc = cc.lower()
    
    if(cc == 'y'):
        numSteps = int(input("input sample batch size: "))
        stepSize = int(input("input inter-batch step size: "))
        trainSplit = float(input("input train/test split ratio (0-1): "))
    else:
        numSteps = WISDM_BATCH_SIZE_DEF
        stepSize = WISDM_STEP_SIZE_DEF
        trainSplit = WISDM_TRAIN_RATIO_DEF
    
    return numSteps, stepSize, trainSplit

def convertModelDirectory(inModelDir, savedName = 'model'):
    converter = tf.lite.TFLiteConverter.from_saved_model(inModelDir)
    tfLiteModel = converter.convert()
    
    with open(savedName, 'wb') as f:
         f.write(tfLiteModel)

def convertModelKeras(inModel, savedName = 'model'):
    converter = tf.lite.TFLiteConverter.from_keras_model(inModel)
    tfLiteModel = converter.convert()
    
    with open(savedName, 'wb') as f:
         f.write(tfLiteModel)

# def testModel(inModel, numRepeats, epochs = 10, batch_size = 32):
#     if(inModel = "")
# trainX, trainY, testX, testY = dp.loadHARSet()
# print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
# print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
# scores = []
# testModel = mo.testCNN(trainX, trainY, testX, testY)
# print(testModel.summary())
# for i in range(5):
#     testModel = mo.testCNN(trainX, trainY, testX, testY)
#     testModel.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
#     _, score = testModel.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    
#     score = score * 100.0
#     print('>#%d: %.3f' % (i+1, score))
#     if(i == 4):
#         testModel.save("CNN_test_UCI")
#     scores.append(score)


# trainX, trainY, testX, testY = dp.loadWISDM(mode="at", n_steps=200, step=20, trainSplit=0.2)
# print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
# print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
# testModel = mo.testCNN(trainX, trainY, testX, testY)
# print(testModel.summary())
# for i in range(5):
    
    
#     testModel = mo.testCNN(trainX, trainY, testX, testY)
#     testModel.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
#     _, score = testModel.evaluate(testX, testY, batch_size=batch_size, verbose=verbose)
    
#     score = score * 100.0
#     print('>#%d: %.3f' % (i+1, score))
#     if(i == 4):
#         testModel.save("CNN_test_WISDM")
#     scores.append(score)

# print(scores)
# m, s = npy.mean(scores), npy.std(scores)
# print('Accuracy: %.3f%% (+/-%.3f)' % (m,s))

modelType = "CNN_test"
dataSet = "UCI_HAR"
menu = ''
model = 0
trainX, trainY, testX, testY = 0, 0, 0, 0
scores = []

print("-----------------------------------")

while(menu != 'x' and menu != 'X'):
    print("Current Model selected: "+ modelType)
    print("Current Dataset slected: " + dataSet)
    print("Please choose a menu option (x to quit):\n 1 Select Model\n 2 Select Dataset\n 3 train and save model\n 4 evaluate model")
    
    menu = input("> ")
    if(menu == '1'):
        print("1. CNN\n2. RNN\n3. CNN_test")
        menu = input("input the number for the model you wish to use\n> ")
        if(menu == '1'):
            modelType = "CNN"
        elif(menu == '2'):
            modelType = "RNN"
        elif(menu == '3'):
            modelType = "CNN_test"
    elif(menu == '2'):
        dp.checkAllSets()
        menu = input("input the number for the database you wish to use\n> ")
        if(menu == '1'):
            dataSet = "WISDM_ar"
            numSteps, stepSize, trainSplit = configWISDM()
            print("loading WISDM_ar dataset...")
            trainX, trainY, testX, testY = dp.loadWISDM(mode="ar", n_steps=numSteps, step=stepSize, trainSplit=trainSplit)
            print("Successfully loaded!")
        elif(menu == '2'):
            dataSet = "WISDM_at"
            numSteps, stepSize, trainSplit = configWISDM()
            print("loading WISDM_at dataset...")
            trainX, trainY, testX, testY = dp.loadWISDM("at", n_steps=numSteps, step=stepSize, trainSplit=trainSplit)
            print("Successfully loaded!")
        elif(menu == '3'):
            dataSet = "UCI_HAR"
            print("loading UCI HAR dataset...")
            trainX, trainY, testX, testY = dp.loadHARSet()
            print("Successfully loaded!")
        
        print("Number of Signal Batches: " + str(trainX.shape[0]))
    elif(menu == '3'):
        print("Current settings:\nepochs -> " + str(epochs) + 
              "\nBatch Size -> " + str(batch_size) + 
              "\nVerbosity -> " + str(verbose))
        menuSub = input("reconfigure model fitting settings?\n(y/n)> ")
        menuSub = menuSub.lower()
        
        if(menuSub == 'y'):
            epochs = int(input("#training epochs: "))
            batch_size = int(input("model batch size: "))
            verbose = int(input("training verbosity(0,1,2): "))
        
        if(modelType == "CNN"):
            model = mo.CNNModel(trainX, trainY)
        elif(modelType == "RNN"):
            model = mo.RNNModel(trainX, trainY)
        elif(modelType == "CNN_test"):
            model = mo.testCNN(trainX, trainY)
        
        print("fitting dataset to model...")
        print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
        print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
        if(modelType == "CNN"):
            model.fit([trainX,trainX,trainX], trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
            _, score = model.evaluate([testX,testX,testX], testY, batch_size=batch_size, verbose=0)
        else:
           model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
           _, score = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        
        score = score * 100.0
        
        modelName = modelType + "-" + dataSet
        model.save("saved/" + modelName)
        
        descFile = open("saved/tflite/" + modelName + "_readme.txt", "w")
        descFile.write("---MODEL STATISTICS---")
        descFile.write("\r Model Type: " + modelType)
        descFile.write("\r trained using: " + dataSet)
        descFile.write("\r Signal Batch Size: " + str(trainX.shape[1]))
        descFile.write("\r Number of features per signal: " + str(trainX.shape[2]))
        if(dataSet == "UCI_HAR"):
           descFile.write("\r Signal Format: [grav_accel_x, grav_accel_y, grav_accel_z, non_grav_accel_x, non_grav_accel_y, non_grav_accel_z, gyro_x, gyro_y, gyro_z]")
           descFile.write("\r Output Format: [Walking, Walking up, Walking down, sitting, standing, lying]")
        elif("WISDM" in dataSet):
            descFile.write("\r Signal Format: [grav_accel_x, grav_accel_y, grav_accel_z]")
            descFile.write("\r Output Format: [Downstairs, Jogging, Sitting, Standing, Upstairs, Walking]")
        
        descFile.write("\r\r---TRAINING STATISTICS---")
        descFile.write("\r Epochs: " + str(epochs))
        descFile.write("\r Training Batch Size: " + str(batch_size))
        descFile.write("\r Evaluation score: %.3f" % (score))
        descFile.close()
        convertModelKeras(model, "saved/tflite/" + modelName + ".tflite")
        
        print(model.summary())
        print("fitting complete!")
    elif(menu == '4'):
        print("running model test")
        if(modelType == "CNN"):
            _, score = model.evaluate([testX,testX,testX], testY, batch_size=batch_size, verbose=0)
        else:
           _, score = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
        
        
        score = score * 100.0
        print('test accuracy: %.3f' % (score))
