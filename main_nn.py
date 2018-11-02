import pandas as pd
from keras.models import Input

from models.nn_cnn import create_cnn
from models.nn_ensemble import create_ensemble
from models.nn_mlpc import create_mlpc
from utils import prepare_data, compile_and_train, evaluate_accuracy, visualize_history

batch_size = 256
num_classes = 10
epochs = 1

# Load data
data_train = pd.read_csv('dataset/fashion-mnist_train.csv')
data_test = pd.read_csv('dataset/fashion-mnist_test.csv')

# Prepare data
input_shape, X_train, X_val, y_train, y_val, X_test, y_test = prepare_data(data_train, data_test)

# define model input
model_input = Input(shape=input_shape)

# Initialize MLPC model and train
model_mlpc = create_mlpc(model_input, num_classes)
history_mlpc = compile_and_train(model_mlpc, X_train, y_train, X_val, y_val, batch_size, num_epochs=epochs)
evaluate_accuracy(model_mlpc, X_test, y_test)

# instantiate CNN model and train
model_cnn = create_cnn(model_input, num_classes)
history_cnn = compile_and_train(model_cnn, X_train, y_train, X_val, y_val, batch_size, num_epochs=epochs)
evaluate_accuracy(model_cnn, X_test, y_test)

# Create ensemble of MLPC and CNN
models = [model_mlpc, model_cnn]
model_ensemble = create_ensemble(models, model_input)
history_ensemble = compile_and_train(model_ensemble, X_train, y_train, X_val, y_val, batch_size, num_epochs=epochs)
evaluate_accuracy(model_ensemble, X_test, y_test)

# Visualize histories
visualize_history(history_mlpc)
visualize_history(history_cnn)
visualize_history(history_ensemble)
