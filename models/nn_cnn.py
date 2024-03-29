from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Input


def create_cnn_ske(input_shape, num_classes):
    """
    define cnn model

    :param input_shape:
    :param num_classes:
    :return:
    """
    model_input = Input(shape=input_shape)
    cnn_out, x = create_cnn_layers_ske(model_input, num_classes)
    model = Model(model_input, x, name='cnn')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_cnn_layers_ske(model_input, num_classes, out_name=''):
    x = Conv1D(39, kernel_size=3, activation='relu', padding="same", data_format="channels_first",
               kernel_initializer='he_normal')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=17, kernel_size=3, padding="same", data_format="channels_first", activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    cnn_out = Dropout(0.5)(x)
    x = Dense(128)(cnn_out)
    x = LeakyReLU()(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(x)
    return cnn_out, x


def create_cnn_iner(input_shape, num_classes):
    """
    define cnn model

    :param input_shape:
    :param num_classes:
    :return:
    """
    model_input = Input(shape=input_shape)
    cnn_out, x = create_cnn_layers_iner(model_input, num_classes)
    model = Model(model_input, x, name='cnn')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    return model


def create_cnn_layers_iner(model_input, num_classes, out_name=''):
    x = Conv1D(105, kernel_size=3, activation='relu', padding="same", data_format="channels_first",
               kernel_initializer='he_normal')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=50, kernel_size=3, padding="same", data_format="channels_first", activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    cnn_out = Dropout(0.5)(x)
    x = Dense(64)(cnn_out)
    x = LeakyReLU()(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(x)
    return cnn_out, x
