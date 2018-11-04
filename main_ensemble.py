# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os
import sys

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

from common.dataprep import definitions, get_dataset
from models.ensemble_lstm_cnn import create_lstm_cnn_ensemble


def visualize_history(history, prefix='', plot_loss=False, show=True):
    accuracy = history.history['main_output_acc']
    val_accuracy = history.history['val_main_output_acc']
    loss = history.history['main_output_loss']
    val_loss = history.history['val_main_output_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title(prefix + 'Training and validation accuracy')
    plt.legend()
    if plot_loss:
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(prefix + 'Training and validation loss')
        plt.legend()
    if show:
        plt.show()


def run(plot=False):
    """
    Model creation
    """
    num_classes = 28
    epochs = 200
    input_shape_iner = (107, 6)
    input_shape_ske = (41, 60)

    avg_val_acc = 0
    avg_mae = 0
    avg_loss = 0
    hists = []

    for i in range(5):
        X_iner = trainX_iner[i]
        X_ske = trainX_ske[i]
        model = create_lstm_cnn_ensemble(input_shape_iner, input_shape_ske, num_classes)
        if i == 0:
            print(model.summary())
        hist = model.fit([X_iner, X_ske], trainY_ske[i], validation_data=(
            [testX_iner[i], testX_ske[i]], testY_ske[i]),
                         callbacks=[EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')],
                         epochs=epochs)
        hists.append(hist)
        avg_mae += hist.history['val_mean_absolute_error'][-1]
        avg_loss += hist.history['val_loss'][-1]
        avg_val_acc += hist.history['val_acc'][-1]

    print("average mae : " + str(avg_mae / 5))
    print("average loss : " + str(avg_loss / 5))
    print("average accuracy: " + str(avg_val_acc / 5))

    print("\n\nEvaluation Summary")
    for i in range(5):
        hist = hists[i]
        if i > 0:
            plt.figure()
        visualize_history(hist, 'multimodal_%d-' % i, show=False)

    if plot:
        plt.show()


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
