# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os
import sys

import numpy as np
from keras.models import Input

from common.dataprep import definitions, get_dataset
from models.nn_mlpc import create_mlpc
from common.utils import model_train_early_stop, visualize_history, evaluate_accuracy

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
Model configuration
"""
batch_size = 32
epochs = 200
num_classes = 28
model_input_iner = Input(shape=(107, 6))
model_iner = create_mlpc(model_input_iner, num_classes)

model_input_ske = Input(shape=(41, 60))
model_ske = create_mlpc(model_input_ske, num_classes)

"""
Model training and evaluation
"""
VISUALIZATION = False
avg_val_acc_ske = 0
avg_loss_ske = 0
avg_val_acc_iner = 0
avg_loss_iner = 0

for i in range(5):
    X_train_iner = trainX_iner[i]
    y_train_iner = trainY_iner[i]
    X_test_iner = testX_iner[i]
    y_test_iner = testY_iner[i]

    X_train_ske = trainX_ske[i]
    y_train_ske = trainY_ske[i]
    X_test_ske = testX_ske[i]
    y_test_ske = testY_ske[i]

    hist_iner = model_train_early_stop(model_iner, X_train_iner, y_train_iner, X_test_iner, y_test_iner,
                                       batch_size, num_epochs=epochs)

    hist_ske = model_train_early_stop(model_ske, X_train_ske, y_train_ske, X_test_ske, y_test_ske,
                                      batch_size, num_epochs=epochs)

    if VISUALIZATION:
        visualize_history(hist_iner, 'inertial_%d-' % i, plot_loss=False)
        visualize_history(hist_ske, 'skeleton_%d-' % i, plot_loss=False)

    avg_loss_ske += hist_ske.history['val_loss'][-1]
    avg_val_acc_ske += hist_ske.history['val_acc'][-1]

    avg_loss_iner += hist_iner.history['val_loss'][-1]
    avg_val_acc_iner += hist_iner.history['val_acc'][-1]

print("ske average loss: " + str(avg_loss_ske / 5))
print("ske average accuracy: " + str(avg_val_acc_ske / 5))
print("iner average loss: " + str(avg_loss_iner / 5))
print("iner average accuracy: " + str(avg_val_acc_iner / 5))

print("\n\nEvaluation Summary")
for i in range(5):
    X_train_iner = trainX_iner[i]
    y_train_iner = trainY_iner[i]
    X_test_iner = testX_iner[i]
    y_test_iner = testY_iner[i]

    X_train_ske = trainX_ske[i]
    y_train_ske = trainY_ske[i]
    X_test_ske = testX_ske[i]
    y_test_ske = testY_ske[i]

    print("\nSubject index: " + str(i))
    acc_iner = evaluate_accuracy(model_iner, X_test_iner, y_test_iner)
    acc_ske = evaluate_accuracy(model_ske, X_test_ske, y_test_ske)
    print("::iner accuracy: " + str(acc_iner))
    print("::ske accuracy: " + str(acc_ske))
