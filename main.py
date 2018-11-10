# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""
from common.dataprep import prepare_data
import main_nn_mlpc
import main_nn_cnn
import main_lstm
import main_lstm_ensemble
import main_cnn_lstm
import main_ensemble


def main():
    trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner = prepare_data()

    print("\n\n:::MLPC model:::")
    main_nn_mlpc.run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner)

    print("\n\n:::CNN model:::")
    main_nn_cnn.run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner)

    print("\n\n:::LSTM model:::")
    main_lstm.run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner)

    print("\n\n:::LSTM ensemble model:::")
    main_lstm_ensemble.run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner,
                           testY_iner)

    print("\n\n:::Hybrid CNN + LSTM model:::")
    main_cnn_lstm.run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner)

    print("\n\n:::CNN + LSTM ensemble model:::")
    main_ensemble.run(trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_iner, trainY_iner, testX_iner, testY_iner)


if __name__ == "__main__":
    main()
