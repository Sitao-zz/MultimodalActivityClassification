import numpy as np
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.models import Sequential


def create_lstm_simple(input_shape, num_class, num_neuros, num_dp):
    model = Sequential()
    model.add(LSTM(num_neuros, return_sequences=True, input_shape=input_shape,
                   dropout=0.3, recurrent_dropout=0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(num_dp))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def create_lstm_layers_ske(skeleton_input, input_shape_ske, num_classes, out_name=''):
    ske_lstm_1 = LSTM(128, input_shape=input_shape_ske, return_sequences=True)(skeleton_input)
    ske_lstm_out = LSTM(256, return_sequences=False)(ske_lstm_1)
    ske_dense = Dense(128)(ske_lstm_out)
    skeleton_out = Dense(units=num_classes, name=out_name)(ske_dense)
    return ske_lstm_out, skeleton_out


def create_lstm_layers_iner(inertial_input, input_shape_iner, num_classes, out_name=''):
    iner_lstm_1 = LSTM(50, input_shape=input_shape_iner, return_sequences=True)(inertial_input)
    iner_lstm_out = LSTM(100, return_sequences=False)(iner_lstm_1)
    iner_dense = Dense(50)(iner_lstm_out)
    inertial_out = Dense(units=num_classes, name=out_name)(iner_dense)
    return iner_lstm_out, inertial_out
