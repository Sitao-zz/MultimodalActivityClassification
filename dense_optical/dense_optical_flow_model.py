from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import np_utils
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from common.dataprep import definitions
from models.modeldef import ResearchModels
import pandas as pd
import numpy as np
import pickle as pk
import time

import tensorflow as tf
from keras import backend as K


def train(data, model, seq_length, features, classes,
          saved_model=None, concat=False):
    # Set variables.
    nb_epoch = 300
    batch_size = 32

    concat = False

    f1_results, precision_results, recall_results = [], [], []

    for k in range(5):
        y, y_val = [], []
        X = np.ndarray(shape=(len(trainsets[k]), seq_length, features), dtype=np.float64)
        X_val = np.ndarray(shape=(len(validationsets[k]), seq_length, features), dtype=np.float64)
        t_index, v_index = 0, 0
        for row in data.itertuples():
            # print("loading: "+ row.Index)
            sequence = row.hof
            if concat:
                # We want to pass the sequence back as a single array. This
                # is used to pass into a CNN or MLP, rather than an RNN.
                sequence = np.concatenate(sequence).ravel()
            yencode = np_utils.to_categorical(classes.index(str(row.target)), len(classes))[0]  # One-hot encoding
            if row.Index in trainsets[k]:
                X[t_index] = sequence
                y.append(yencode)
                t_index = t_index + 1
            elif row.Index in validationsets[k]:
                X_val[v_index] = sequence
                y_val.append(yencode)
                v_index = v_index + 1

        y = np.array(y)
        y_val = np.array(y_val)

        # Get the model.
        rm = ResearchModels(len(classes), model, seq_length, features, saved_model)

        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath='./data/checkpoints/' + model + '_hof' + str(k) + '.hdf5',
            verbose=1,
            save_best_only=True)

        # Helper: TensorBoard
        tb = TensorBoard(log_dir='./data/logs')

        # Helper: Stop when we stop learning.
        early_stopper = EarlyStopping(patience=50)

        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger('./data/logs/' + model + '_hof' + str(k) + '-training-' + \
                               str(timestamp) + '.log')

        # Configure GPU
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=config)
        K.set_session(sess)

        # Fit
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            epochs=nb_epoch)

        results = rm.model.predict(X_val, batch_size=batch_size, verbose=1)
        actual = [int(i) for i, j in zip(data.target, data.index) if j in validationsets[k]]
        pred = [int(classes[int(np.argmax(i))]) for i in results]

        # Save predictions to csv for ensemble scoring later
        prediction_output = pd.DataFrame()
        prediction_output['Predictions'] = pred
        prediction_output.to_csv('./results/predictions_hof' + str(k) + '.csv')

        report = classification_report(actual, pred, classes, digits=2)
        f1 = f1_score(actual, pred, classes, average='macro')
        precision = precision_score(actual, pred, classes, average='macro')
        recall = recall_score(actual, pred, classes, average='macro')
        f1_results.append(f1)
        precision_results.append(precision)
        recall_results.append(recall)
        print(report)
        print(f1)
        text_file = open("./results/report_hof" + str(k) + ".txt", "w")

        text_file.write(report)
        text_file.close()

        K.clear_session()

    avgf1 = np.mean(np.array(f1_results))
    avgprecision = np.mean(np.array(precision_results))
    avgrecall = np.mean(np.array(recall_results))
    print('Average precision score: {0:0.3f}'.format(avgprecision))
    print('Average recall score: {0:0.3f}'.format(avgrecall))
    print('Average F1 score: {0:0.3f}'.format(avgf1))


def rescale(input_list, size):
    skip = len(input_list) // size
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    return output[:size]


if __name__ == '__main__':

    dataset, trainsets, validationsets = definitions()

    # Load dataset
    f = open("./hofset_48_np.pk", "rb")
    hofset = pk.load(f)
    f.close()

    f = open("./all_df.pk", "rb")
    all_data = pk.load(f)
    f.close()
    hofdf = pd.DataFrame()

    maxlength = min([len(i) for i in hofset])
    for i in range(len(hofset)):
        hofset[i] = np.array(rescale(list(hofset[i]), maxlength))

    # hofset = [ rescale(i, maxlength) for i in hofset ])
    hofdf['hof'] = hofset
    hofdf.index = all_data.index
    classes = []
    for item in all_data.itertuples():
        if item[-1] not in classes:
            classes.append(item[-1])
    classes = sorted(classes)

    model = 'lstm_hof'
    saved_model = None  # None or weights file
    seq_length = maxlength

    features = len(hofdf['hof'][0][0])

    hofdf['target'] = all_data.target

    # MLP requires flattened features.
    if model == 'mlp':
        concat = True
    else:
        concat = False

    train(hofdf, model, seq_length, features, classes, saved_model=saved_model, concat=concat)
