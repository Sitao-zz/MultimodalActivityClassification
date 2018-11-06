from keras.layers import Dense, Dropout, Flatten
from keras.models import Model, Input


def create_mlpc(input_shape, num_classes):
    """
    define mlpc model

    :param model_input:
    :param num_classes:
    :return:
    """
    model_input = Input(shape=input_shape)
    mlpc_out, x = create_mlpc_layers(model_input, num_classes)
    model = Model(model_input, x, name='mplc')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_mlpc_layers(model_input, num_classes, out_name=''):
    x = Flatten()(model_input)
    x = Dense(128, activation='relu')(x)
    mlpc_out = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax', name=out_name)(mlpc_out)
    return mlpc_out, x
