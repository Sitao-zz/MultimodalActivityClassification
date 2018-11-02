from matplotlib import pyplot as plt

import common.dataprep_iner as cd
from models.lstm_simple import create_lstm_simple

X_train, X_test, Y_train, Y_test, numClass = cd.definitions()

from keras.callbacks import EarlyStopping

# Create the model
model = create_lstm_simple(numClass, 500, 0.01)

# Train model
history = model.fit(X_train, Y_train, callbacks=[EarlyStopping(monitor='acc', patience=10, verbose=1, mode='auto')],
                    epochs=30, batch_size=100)

# Evaluate model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

import seaborn as sns

sns.set(style="darkgrid")
plt.plot(history.history['acc'])
plt.title('RNN LSTM - Activity Classification')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
