from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model


def create_cnn(model_input, num_classes):
    """
    define cnn model

    :param model_input:
    :param num_classes:
    :return:
    """
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
               kernel_initializer='he_normal')(model_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(model_input, x, name='cnn')
    print(model.summary())
    return model
