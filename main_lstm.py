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
from common.utils import evaluate_accuracy
from models.lstm_simple import create_lstm_simple

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
batch_size = 32
epochs = 2
numClass = 28
input_shape = (107, 6)
model = create_lstm_simple(input_shape, numClass, 500, 0.01)

"""
Model training and evaluation
"""
avg_val_acc = 0
avg_loss = 0

for i in range(5):
    X_iner = trainX_iner[i]
    hist = model.fit(X_iner, trainY_iner[i], validation_data=(testX_iner[i], testY_iner[i]),
                     callbacks=[EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')],
                     epochs=epochs, batch_size=batch_size)

    avg_loss += hist.history['val_loss'][-1]
    avg_val_acc += hist.history['val_acc'][-1]

print("average loss : " + str(avg_loss / 5))
print("average accuracy: " + str(avg_val_acc / 5))