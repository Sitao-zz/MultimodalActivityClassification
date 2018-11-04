# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os
import sys

from common.dataprep import definitions, get_dataset
import main_cnn_lstm
import main_nn_cnn
import main_ensemble
import main_lstm
import main_lstm_ensemble
import main_nn_mlpc

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR + "/dataset")

"""
Data preparation
"""
dataset, trainsets, validationsets = definitions()
trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner = get_dataset(trainsets,
                                                                                                             validationsets)

print("\n\n:::MLPC model:::")
main_nn_mlpc.run()
print("\n\n:::CNN model:::")
main_nn_cnn.run()
print("\n\n:::LSTM model:::")
main_lstm.run()
print("\n\n:::LSTM ensemble model:::")
main_lstm_ensemble.run()
print("\n\n:::Hybrid CNN + LSTM model:::")
main_cnn_lstm.run()
print("\n\n:::CNN + LSTM ensemble model:::")
main_ensemble.run()
