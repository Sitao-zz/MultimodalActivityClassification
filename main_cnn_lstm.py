# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os
import sys
import matplotlib.pyplot as plt

from common.dataprep import prepare_data
from common.utils import model_train_early_stop, visualize_history
from models.cnn_lstm import create_cnn_lstm_iner, create_cnn_lstm_ske

from common.utils import evaluate_classification


def run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner):
    """
    Model configuration
    """
    batch_size = 32
    epochs = 200
    num_classes = 27
    input_shape_iner = (107, 6)
    input_shape_ske = (41, 60)

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

        model_iner = create_cnn_lstm_iner(input_shape_iner, num_classes)
        model_ske = create_cnn_lstm_ske(input_shape_ske, num_classes)
        if i == 0:
            print(model_iner.summary())
            print(model_ske.summary())

        hist_iner = model_train_early_stop(model_iner, X_train_iner, y_train_iner, X_test_iner, y_test_iner,
                                           batch_size, num_epochs=epochs)

        hist_ske = model_train_early_stop(model_ske, X_train_ske, y_train_ske, X_test_ske, y_test_ske,
                                          batch_size, num_epochs=epochs)

        evaluate_classification(model_iner, X_test_iner, y_test_iner, 'cnn-lstm_inertial_%d-Confusion Matrix' % i)
        evaluate_classification(model_ske, X_test_ske, y_test_ske, 'cnn-lstm_skeleton_%d-Confusion Matrix' % i)

        if VISUALIZATION:
            visualize_history(hist_iner, 'inertial_%d-' % i, plot_loss=False)
            visualize_history(hist_ske, 'skeleton_%d-' % i, plot_loss=False)

        print("ske loss [" + str(i) + "]\t" + str(hist_ske.history['val_loss'][-1]))
        print("ske accuracy [" + str(i) + "]\t" + str(hist_ske.history['val_acc'][-1]))
        print("iner loss [" + str(i) + "]\t" + str(hist_iner.history['val_loss'][-1]))
        print("iner accuracy [" + str(i) + "]\t" + str(hist_iner.history['val_acc'][-1]))
        print("\n")

        avg_loss_ske += hist_ske.history['val_loss'][-1]
        avg_val_acc_ske += hist_ske.history['val_acc'][-1]
        avg_loss_iner += hist_iner.history['val_loss'][-1]
        avg_val_acc_iner += hist_iner.history['val_acc'][-1]

    print("ske average loss: " + str(avg_loss_ske / 5))
    print("ske average accuracy: " + str(avg_val_acc_ske / 5))
    print("iner average loss: " + str(avg_loss_iner / 5))
    print("iner average accuracy: " + str(avg_val_acc_iner / 5))
    plt.show()


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT_DIR + "/dataset")
    trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner = prepare_data()
    run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner)
