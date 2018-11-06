from keras.layers import Dense, LSTM, Input, Flatten, Dropout
from keras.models import Model, Sequential


def create_lstm_simple(input_shape, num_classes, num_neuros, num_dp):
    model = Sequential()
    model.add(LSTM(num_neuros, return_sequences=True, input_shape=input_shape,
                   dropout=0.3, recurrent_dropout=0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(num_dp))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def create_lstm_ske(input_shape, num_classes, num_neuros, num_dp):
    model_input = Input(shape=input_shape)
    lstm_out, x = create_lstm_layers_ske(model_input, input_shape, num_classes)
    model = Model(model_input, x, name='lstm_ske')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_lstm_layers_ske(skeleton_input, input_shape_ske, num_classes, out_name=''):
    ske_lstm_1 = LSTM(128, input_shape=input_shape_ske, return_sequences=True)(skeleton_input)
    ske_lstm_out = LSTM(256, return_sequences=False)(ske_lstm_1)
    ske_dense = Dense(128)(ske_lstm_out)
    skeleton_out = Dense(units=num_classes, activation='softmax', name=out_name)(ske_dense)
    return ske_lstm_out, skeleton_out


def create_lstm_iner(input_shape, num_classes, num_neuros, num_dp):
    model_input = Input(shape=input_shape)
    lstm_out, x = create_lstm_layers_iner(model_input, input_shape, num_classes)
    model = Model(model_input, x, name='lstm_iner')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_lstm_layers_iner(inertial_input, input_shape_iner, num_classes, out_name=''):
    iner_lstm_1 = LSTM(50, input_shape=input_shape_iner, return_sequences=True)(inertial_input)
    iner_lstm_out = LSTM(100, return_sequences=False)(iner_lstm_1)
    iner_dense = Dense(50)(iner_lstm_out)
    inertial_out = Dense(units=num_classes, activation='softmax', name=out_name)(iner_dense)
    return iner_lstm_out, inertial_out
