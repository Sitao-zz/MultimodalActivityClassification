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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_cnn_layers_ske(model_input, num_classes, out_name=''):
    x = Conv1D(58, kernel_size=3, activation='relu', padding="same",
               kernel_initializer='he_normal')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=56, kernel_size=3, padding="same", activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    cnn_out = LeakyReLU()(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(cnn_out)
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_cnn_layers_iner(model_input, num_classes, out_name=''):
    x = Conv1D(4, kernel_size=3, activation='relu', padding="same",
               kernel_initializer='he_normal')(model_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=2, kernel_size=3, padding="same", activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    cnn_out = LeakyReLU()(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(cnn_out)
    return cnn_out, x
