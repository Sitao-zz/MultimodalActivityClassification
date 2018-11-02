from keras.layers import Average
from keras.models import Model


def create_ensemble(models, model_input):
    """
    define ensemble model

    :param models:
    :param model_input:
    :return:
    """
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(model_input, y, name='ensemble')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
