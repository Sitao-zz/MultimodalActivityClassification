from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Input


def create_cnn_lstm_ske(input_shape, num_classes):
    """
    define cnn model

    :param input_shape:
    :param num_classes:
    :return:
    """
    model_input = Input(shape=input_shape)
    cnn_out, x = create_cnn_lstm_layers_ske(model_input, num_classes)
    model = Model(model_input, x, name='cnn')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_cnn_lstm_layers_ske(model_input, num_classes, out_name=''):
    x = Conv1D(39, kernel_size=3, activation='relu', padding="same", data_format="channels_first",
               kernel_initializer='he_normal')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=17, kernel_size=3, padding="same", data_format="channels_first", activation='relu')(x)
    x = LSTM(128, input_shape=(17, 60), return_sequences=True)(x)
    x = LSTM(256, return_sequences=False)(x)
    cnn_lstm_out = Dense(128)(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(cnn_lstm_out)
    return cnn_lstm_out, x


def create_cnn_lstm_iner(input_shape, num_classes):
    """
    define cnn model

    :param input_shape:
    :param num_classes:
    :return:
    """
    model_input = Input(shape=input_shape)
    cnn_out, x = create_cnn_lstm_layers_iner(model_input, num_classes)
    model = Model(model_input, x, name='cnn')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_cnn_lstm_layers_iner(model_input, num_classes, out_name=''):
    x = Conv1D(105, kernel_size=3, activation='relu', padding="same", data_format="channels_first",
               kernel_initializer='he_normal')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=50, kernel_size=3, padding="same", data_format="channels_first", activation='relu')(x)
    x = LSTM(50, input_shape=(50, 6), return_sequences=True)(x)
    x = LSTM(100, return_sequences=False)(x)
    cnn_lstm_out = Dense(50)(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(cnn_lstm_out)
    return cnn_lstm_out, x
