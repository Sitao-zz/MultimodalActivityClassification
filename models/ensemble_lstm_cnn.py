import keras.utils
from keras.layers import Dense, Input
from keras.models import Model

from models.lstm_simple import create_lstm_layers_iner
from models.nn_cnn import create_cnn_layers


def create_lstm_cnn_ensemble(input_shape_iner, input_shape_ske, num_classes):
    inertial_input = Input(shape=input_shape_iner, name="iner_lstm_input")
    iner_lstm_out, inertial_out = create_lstm_layers_iner(inertial_input, input_shape_iner, num_classes, "inertial_output")

    skeleton_input = Input(shape=input_shape_ske, name="ske_lstm_input")
    ske_lstm_out, skeleton_out = create_cnn_layers(skeleton_input, num_classes,"skeleton_output")

    merge = keras.layers.concatenate([iner_lstm_out, ske_lstm_out])
    dense_1 = Dense(128, activation='relu')(merge)
    # dense_2 = Dense(128, activation = 'relu')(dense_1)
    main_output = Dense(units=num_classes, activation='softmax', name='main_output')(dense_1)

    model = Model(inputs=[inertial_input, skeleton_input], outputs=main_output)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'acc'])
    return model

