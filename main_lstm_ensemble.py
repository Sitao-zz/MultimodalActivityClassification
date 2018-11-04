# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping

from common.dataprep import definitions, get_dataset
from models.ensemble_lstm import create_lstm_ensemble

np.random.seed(1337)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR + "/dataset")

"""
Data preparation
"""
dataset, trainsets, validationsets = definitions()
trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner = get_dataset(trainsets,
                                                                                                             validationsets)

"""
Model creation 
"""
num_classes = 28
input_shape_iner = (107, 6)
input_shape_ske = (41, 60)

"""
Model training and evaluation
"""
avg_val_acc = 0
avg_mae = 0
avg_loss = 0
avg_val_acc_ske = 0
avg_mae_ske = 0
avg_loss_ske = 0
avg_val_acc_iner = 0
avg_mae_iner = 0
avg_loss_iner = 0

for i in range(5):
    X_iner = trainX_iner[i]
    X_ske = trainX_ske[i]
    model = create_lstm_ensemble(input_shape_iner, input_shape_ske, num_classes)
    hist = model.fit([X_iner, X_ske], [trainY_ske[i], trainY_iner[i], trainY_ske[i]], validation_data=(
        [testX_iner[i], testX_ske[i]], [testY_ske[i], testY_ske[i], testY_ske[i]]),
                     callbacks=[EarlyStopping(monitor='val_main_output_acc', patience=10, verbose=1, mode='auto')],
                     epochs=200)
    avg_mae += hist.history['val_main_output_mean_absolute_error'][-1]
    avg_loss += hist.history['val_main_output_loss'][-1]
    avg_val_acc += hist.history['val_main_output_acc'][-1]
    avg_mae_ske += hist.history['val_skeleton_output_mean_absolute_error'][-1]
    avg_loss_ske += hist.history['val_skeleton_output_loss'][-1]
    avg_val_acc_ske += hist.history['val_skeleton_output_acc'][-1]
    avg_mae_iner += hist.history['val_inertial_output_mean_absolute_error'][-1]
    avg_loss_iner += hist.history['val_inertial_output_loss'][-1]
    avg_val_acc_iner += hist.history['val_inertial_output_acc'][-1]

print("average mae : " + str(avg_mae / 5))
print("average loss : " + str(avg_loss / 5))
print("average accuracy: " + str(avg_val_acc / 5))
print("ske average mae: " + str(avg_mae_ske / 5))
print("ske average loss: " + str(avg_loss_ske / 5))
print("ske average accuracy: " + str(avg_val_acc_ske / 5))
print("iner average mae: " + str(avg_mae_iner / 5))
print("iner average loss: " + str(avg_loss_iner / 5))
print("iner average accuracy: " + str(avg_val_acc_iner / 5))
