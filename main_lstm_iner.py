from matplotlib import pyplot as plt

from common.dataprep_iner import definitions
from models.lstm_simple import create_lstm_simple

X_train, X_test, Y_train, Y_test, num_classes = definitions()

from keras.callbacks import EarlyStopping

# Create the model
input_shape = (326, 6)
model = create_lstm_simple(input_shape, num_classes, 500, 0.01)
print(model.summary())

# Train model
history = model.fit(X_train, Y_train, callbacks=[EarlyStopping(monitor='acc', patience=10, verbose=0, mode='auto')],
                    epochs=30, batch_size=100, verbose=0)

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
