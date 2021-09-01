# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:15:37 2021

@author: zolta
"""

import tensorflow as tf
import numpy as npy
import pandas as pnd
from os import listdir

def checkAllSets(fpb = 'dataset/'):
    
    dSetC = [0,0,0]
    
    for dSet in listdir(fpb):
        if 'HAR' in dSet:
            dSetC[0] = 1
        elif 'WISDM_ar' in dSet:
            dSetC[1] = 1
        elif "WISDM_at" in dSet:
            dSetC[2] = 1
        
    print("WISDM_AR: " + ("found" if dSetC[0] else 'not found'))
    print("WISDM_AT: " + ("found" if dSetC[1] else 'not found'))
    print("UCI HAR: " + ("found" if dSetC[2] else 'not found'))
        

def loadHARSetGroup(group, dSetPath='dataset/UCI HAR Dataset/'):
    XsetPath = dSetPath + group + '/Inertial Signals/'
    filenames = []
    
    #accel with gravity
    filenames.append('total_acc_x_'+group+'.txt')
    filenames.append('total_acc_y_'+group+'.txt')
    filenames.append('total_acc_z_'+group+'.txt')
    
    #raw accel (no gravity)
    filenames.append('body_acc_x_'+group+'.txt')
    filenames.append('body_acc_y_'+group+'.txt')
    filenames.append('body_acc_z_'+group+'.txt')
    
    #gyro
    filenames.append('body_gyro_x_'+group+'.txt')
    filenames.append('body_gyro_y_'+group+'.txt')
    filenames.append('body_gyro_z_'+group+'.txt')
    

    X = []
    Y = []
    
    for fName in filenames:
        X.append(pnd.read_csv(XsetPath + fName, header=None, delim_whitespace=True).values)
    X = npy.dstack(X)
        
    Y = pnd.read_csv(dSetPath + group + '/y_'+group+'.txt', header=None, delim_whitespace=True).values
    return X, Y


def loadHARSet(fpb = 'dataset/UCI HAR Dataset/'):
    a=0
    
    trainX, trainY = loadHARSetGroup('train', fpb)
   
    
    testX, testY = loadHARSetGroup('test', fpb)
   
   
    trainY = trainY -1
    testY = testY -1
    
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
    print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
    
    return trainX, trainY, testX, testY

def loadWISMD_ARset(fpb = 'dataset/'):
    a=0
    
    return a

def loadWISMD_ATset(fpb = 'dataset/'):
    a=0
    
    return a

