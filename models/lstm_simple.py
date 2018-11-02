import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


def create_lstm_simple(activity_count):
    np.random.seed(7)
    model = Sequential()
    model.add(LSTM(200, input_shape=(326, 6)))
    model.add(Dense(activity_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    return model
