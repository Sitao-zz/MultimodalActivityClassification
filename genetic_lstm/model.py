from keras.callbacks import EarlyStopping

from models.lstm_simple import create_lstm_simple


class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test, numClass):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.numClass = numClass

    def evaluate(self, num_neuros, num_dp):
        input_shape = (326, 6)
        model = create_lstm_simple(input_shape, self.numClass, num_neuros, num_dp)
        model.fit(self.X_train, self.Y_train,
                  callbacks=[EarlyStopping(monitor='acc', patience=10, verbose=0, mode='auto')],
                  epochs=50, batch_size=100)
        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        return scores[1]
