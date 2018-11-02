import scipy.io
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Dropout


class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test,numClass):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.numClass=numClass

    def evaluate(self, num_neuros,num_dp):
        np.random.seed(7)
        model = Sequential()
        model.add(LSTM(num_neuros, return_sequences=True, input_shape=(326, 6),
                       dropout=0.3, recurrent_dropout=0.3))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(num_dp))
        model.add(Dense(self.numClass, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        print(model.summary())


        model.fit(self.X_train, self.Y_train,
                  callbacks=[EarlyStopping(monitor='acc', patience=10, verbose=0, mode='auto')],
                  epochs=50, batch_size=100)
        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        return scores[1]
