import keras.utils
from keras.layers import Dense, LSTM, Input
from keras.models import Model


def create_lstm_ensemble():
    inertial_input = Input(shape=(107, 6), name="iner_lstm_input")
    iner_lstm_1 = LSTM(50, input_shape=(107, 6), return_sequences=True)(inertial_input)
    iner_lstm_out = LSTM(100, return_sequences=False)(iner_lstm_1)
    iner_dense = Dense(50)(iner_lstm_out)
    inertial_out = Dense(units=28, name="inertial_output")(iner_dense)
    skeleton_input = Input(shape=(41, 60), name="ske_lstm_input")
    ske_lstm_1 = LSTM(128, input_shape=(41, 60), return_sequences=True)(skeleton_input)
    ske_lstm_out = LSTM(256, return_sequences=False)(ske_lstm_1)
    ske_dense = Dense(128)(ske_lstm_out)
    skeleton_out = Dense(units=28, name="skeleton_output")(ske_dense)
    merge = keras.layers.concatenate([iner_lstm_out, ske_lstm_out])
    dense_1 = Dense(128, activation='relu')(merge)
    # dense_2 = Dense(128, activation = 'relu')(dense_1)
    main_output = Dense(units=28, activation='softmax', name='main_output')(dense_1)
    model = Model(inputs=[inertial_input, skeleton_input], outputs=[main_output, inertial_out, skeleton_out])
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'acc'])
    print(model.summary())
    return model
