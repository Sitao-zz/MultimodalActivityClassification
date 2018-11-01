import scipy.io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
class Model:
    def __init__(self, X_train, Y_train,X_test,Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def evaluate(self, num_neuros, num_epoch):
        np.random.seed(7)
        model = Sequential()
        model.add(LSTM(num_neuros, input_shape=(326, 6)))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        print(model.summary())

        model.fit(self.X_train, self.Y_train,
                            callbacks=[EarlyStopping(monitor='acc', patience=10, verbose=0, mode='auto')],
                            epochs=num_epoch, batch_size=1)
        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        return scores[1]