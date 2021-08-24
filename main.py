# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:37:23 2021

@author: zolta
"""

import tensorflow as tf
import model as mo



class LossHistory(tf.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        

earlyStopping = tf.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
history = LossHistory()