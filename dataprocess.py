# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:15:37 2021

@author: zolta
"""
import numpy as npy
from scipy import stats
import pandas as pnd
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
        
    print(" 1 WISDM_AR: " + ("found" if dSetC[0] else 'not found'))
    print(" 2 WISDM_AT: " + ("found" if dSetC[1] else 'not found'))
    print(" 3 UCI HAR: " + ("found" if dSetC[2] else 'not found'))
        

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
    trainX, trainY = loadHARSetGroup('train', fpb)
    testX, testY = loadHARSetGroup('test', fpb)
   
    trainY = trainY -1
    testY = testY -1
    
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    print("Shape of train Data:\nX: " , trainX.shape , " Y:" , trainY.shape)
    print("Shape of test Data:\nX: ", testX.shape , " Y:" , testY.shape)
    
    return trainX, trainY, testX, testY

def loadWISDM(mode = "ar", fpb = 'dataset/', n_steps = 128, step = 32, trainSplit = 0.25, normalize = False):
    if(mode == 'at'):
        file = fpb + "WISDM_at_v2.0/WISDM_at_v2.0_raw.txt"
    else:
        file = fpb + "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
    
    cols = ['person','activity','timestamp', 'x', 'y', 'z']
    dSet = pnd.read_csv(file, header=None, names = cols)
    dSet['z'] = dSet['z'].str.replace(r';', '')
    dSet = dSet.dropna()
    
    dSetVals = []
    label = []

    for i in range(0, len(dSet) - n_steps, step):
        x = dSet['x'].values[i: i + n_steps]
        y = dSet['y'].values[i: i + n_steps]
        z = dSet['z'].values[i: i + n_steps]
        x= npy.asarray(x, dtype = npy.float32)/20
        y= npy.asarray(z, dtype = npy.float32)/20
        z= npy.asarray(y, dtype = npy.float32)/20
        dSetVals.append([x,y,z])
        l = stats.mode(dSet['activity'][i: i+ n_steps])[0][0]
        label.append(l)
        
    dSetVals = npy.transpose(npy.asarray(dSetVals, dtype= npy.float32),(0,2,1))
    label = npy.asarray(pnd.get_dummies(label), dtype = npy.float32)
    
    
    trainX, testX, trainY, testY =train_test_split(dSetVals, label, test_size = trainSplit, random_state=69)
    return trainX, trainY, testX, testY

