import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.layers import LSTM
from keras.models import Sequential


def create_lstm_simple(input_shape, num_class, num_neuros, num_dp):
    np.random.seed(7)
    model = Sequential()
    model.add(LSTM(num_neuros, return_sequences=True, input_shape=input_shape,
                   dropout=0.3, recurrent_dropout=0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(num_dp))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print(model.summary())
    return model
