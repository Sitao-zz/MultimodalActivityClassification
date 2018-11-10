import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

class_names = (
"swipt_left      ", "swipt_right     ", "wave            ", "clap            ", "throw           ", "arm_cross       ",
"basketball_shoot", "draw_x          ", "draw_circle_CW  ", "draw_circle_CCW ", "draw_triangle   ", "bowling         ",
"boxing          ", "baseball_swing  ", "tennis_swing    ", "arm_curl        ", "tennis_serve    ", "push            ",
"knock           ", "catch           ", "pickup_throw    ", "jog             ", "walk            ", "sit2stand       ",
"stand2sit       ", "lunge           ", "squat           ")


def model_train(model, X_train, y_train, X_val, y_val, batch_size, num_epochs):
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=0,
                        validation_data=(X_val, y_val))
    return history


def model_train_early_stop(model, X_train, y_train, X_val, y_val, batch_size, num_epochs):
    history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=0,
                        validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping(monitor='val_acc', patience=10, verbose=0,
                                                 mode='auto')])
    return history


def evaluate_accuracy(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]


def evaluate_classification(model, X_test, y_test, title=""):
    predictions = model.predict(X_test)
    y_pred = (predictions > 0.5)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(16, 12), dpi=60)
    plot_confusion_matrix(matrix, class_names)
    plt.title(title)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()


def visualize_history(history, prefix='', plot_loss=False):
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title(prefix + 'Training and validation accuracy')
    plt.legend()
    if plot_loss:
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title(prefix + 'Training and validation loss')
        plt.legend()
    plt.show()
