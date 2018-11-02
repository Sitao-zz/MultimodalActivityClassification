from keras.layers import Dense, Dropout, Flatten
from keras.models import Model


def create_mlpc(model_input, num_classes):
    """
    define mlpc model

    :param model_input:
    :param num_classes:
    :return:
    """
    x = Dense(128, activation='relu')(model_input)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model_input, x, name='mplc')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
