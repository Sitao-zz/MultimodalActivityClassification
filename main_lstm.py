# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os
import sys

from keras.callbacks import EarlyStopping

from common.dataprep import definitions, get_dataset
from models.lstm_simple import create_lstm_simple


def run():
    """
    Model creation
    """
    batch_size = 32
    epochs = 200
    numClass = 28
    input_shape_iner = (107, 6)
    input_shape_ske = (41, 60)

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

        model_iner = create_lstm_simple(input_shape_iner, numClass, 6, 0.01)
        model_ske = create_lstm_simple(input_shape_ske, numClass, 60, 0.01)
        if i == 0:
            print(model_iner.summary())
            print(model_ske.summary())

        hist_iner = model_iner.fit(X_train_iner, y_train_iner, validation_data=(X_test_iner, y_test_iner),
                         callbacks=[EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')],
                         epochs=epochs, batch_size=batch_size, verbose=0)
        hist_ske = model_ske.fit(X_train_ske, y_train_ske, validation_data=(X_test_ske, y_test_ske),
                         callbacks=[EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')],
                         epochs=epochs, batch_size=batch_size, verbose=0)

        print("ske loss [" + str(i) + "]" + str(hist_ske.history['val_loss'][-1]))
        print("ske accuracy [" + str(i) + "]" + str(hist_ske.history['val_acc'][-1]))
        print("iner loss [" + str(i) + "]" + str(hist_iner.history['val_loss'][-1]))
        print("iner accuracy [" + str(i) + "]" + str(hist_iner.history['val_acc'][-1]))
        print("\n")

        avg_loss_iner += hist_iner.history['val_loss'][-1]
        avg_val_acc_iner += hist_iner.history['val_acc'][-1]
        avg_loss_ske += hist_ske.history['val_loss'][-1]
        avg_val_acc_ske += hist_ske.history['val_acc'][-1]

    print("iner average loss : " + str(avg_loss_iner / 5))
    print("iner average accuracy: " + str(avg_val_acc_iner / 5))
    print("ske average loss : " + str(avg_loss_ske / 5))
    print("ske average accuracy: " + str(avg_val_acc_ske / 5))


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR + "/dataset")

"""
Data preparation
"""
dataset, trainsets, validationsets = definitions()
trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner = get_dataset(trainsets,
                                                                                                             validationsets)

"""
Model training and evaluation
"""
run()
