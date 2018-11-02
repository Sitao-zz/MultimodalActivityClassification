import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def prepare_data(data_train, data_test):
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    X = np.array(data_train.iloc[:, 1:])
    y = to_categorical(np.array(data_train.iloc[:, 0]))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
    # Test data
    X_test = np.array(data_test.iloc[:, 1:])
    y_test = to_categorical(np.array(data_test.iloc[:, 0]))
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_test /= 255
    X_val /= 255
    return input_shape, X_train, X_val, y_train, y_val, X_test, y_test


def compile_and_train(model, X_train, y_train, X_val, y_val, batch_size, num_epochs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=1,
                        validation_data=(X_val, y_val))
    return history


def compile_and_train_early_strop(model, X_train, y_train, X_val, y_val, batch_size, num_epochs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping(monitor='val_acc', patience=10, verbose=1,
                                                 mode='auto')])
    return history


def evaluate_accuracy(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]


def visualize_history(history, prefix=''):
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title(prefix + 'Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(prefix + 'Training and validation loss')
    plt.legend()
    plt.show()
