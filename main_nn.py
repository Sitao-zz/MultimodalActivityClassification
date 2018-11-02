# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os

import numpy as np
import pandas as pd
import scipy.io as scio
from keras.models import Input

np.random.seed(1337)

# %%
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR + "/dataset")
import dataprep as cd

dataset, trainsets, validationsets = cd.definitions()


# %%
def pre_processing_inertial(data):  # (200,6)
    predata = pd.DataFrame(data).ewm(span=3).mean().values  # convert the dataframe into numpy array
    if data.shape[0] > 107:
        x = data.shape[0] // 2
        predata = predata[(x - 53):(x + 54), :]

    return predata


# %%
def pre_processing_skeleton(data):  # (200,6)
    predata = data
    if data.shape[2] > 41:
        x = data.shape[2] // 2
        predata = predata[:, :, (x - 20):(x + 21)]
    return predata


# %%
# np.where(data_inertial[1] == max(data_inertial[1]))
# %%
trainX_skeleton = [[], [], [], [], []]
trainY_skeleton = [[], [], [], [], []]
testX_skeleton = [[], [], [], [], []]
testY_skeleton = [[], [], [], [], []]
trainX_inertial = [[], [], [], [], []]
trainY_inertial = [[], [], [], [], []]
testX_inertial = [[], [], [], [], []]
testY_inertial = [[], [], [], [], []]
for i in range(5):
    for j in trainsets[i]:
        dic_mat_inertial = scio.loadmat("dataset/Inertial/" + j + "_inertial.mat")
        dic_mat_skeleton = scio.loadmat("dataset/Skeleton/" + j + "_skeleton.mat")
        data_skeleton = dic_mat_skeleton["d_skel"]
        data_inertial = dic_mat_inertial["d_iner"]
        # data = data[:,:,:41]
        data_skeleton = pre_processing_skeleton(data_skeleton)
        data_inertial = pre_processing_inertial(data_inertial)

        traindata_skeleton = np.reshape(data_skeleton, (60, 41))
        inputdata_skeleton = traindata_skeleton.T
        inputdata_skeleton = np.ndarray.tolist(inputdata_skeleton)
        trainX_skeleton[i].append(inputdata_skeleton)
        trainY_skeleton[i].append(int(j.split('_')[0][1:]))

        inputdata_inertial = pre_processing_inertial(data_inertial)
        inputdata_inertial = np.ndarray.tolist(inputdata_inertial)
        trainX_inertial[i].append(inputdata_inertial)
        trainY_inertial[i].append(int(j.split('_')[0][1:]))

    for j in validationsets[i]:
        dic_mat_inertial = scio.loadmat("dataset/Inertial/" + j + "_inertial.mat")
        dic_mat_skeleton = scio.loadmat("dataset/Skeleton/" + j + "_skeleton.mat")
        data_skeleton = dic_mat_skeleton["d_skel"]
        data_inertial = dic_mat_inertial["d_iner"]

        data_skeleton = pre_processing_skeleton(data_skeleton)
        data_inertial = pre_processing_inertial(data_inertial)

        traindata_skeleton = np.reshape(data_skeleton, (60, 41))
        inputdata_skeleton = traindata_skeleton.T
        inputdata_skeleton = np.ndarray.tolist(inputdata_skeleton)
        testX_skeleton[i].append(inputdata_skeleton)
        testY_skeleton[i].append(int(j.split('_')[0][1:]))

        inputdata_inertial = pre_processing_inertial(data_inertial)
        inputdata_inertial = np.ndarray.tolist(inputdata_inertial)
        testX_inertial[i].append(inputdata_inertial)
        testY_inertial[i].append(int(j.split('_')[0][1:]))

# %%
import sklearn.preprocessing

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(28))
for i in range(5):
    trainX_skeleton[i] = np.array(trainX_skeleton[i])
    testX_skeleton[i] = np.array(testX_skeleton[i])
    trainY_skeleton[i] = label_binarizer.transform(trainY_skeleton[i])
    testY_skeleton[i] = label_binarizer.transform(testY_skeleton[i])

    trainX_inertial[i] = np.array(trainX_inertial[i])
    testX_inertial[i] = np.array(testX_inertial[i])
    trainY_inertial[i] = label_binarizer.transform(trainY_inertial[i])
    testY_inertial[i] = label_binarizer.transform(testY_inertial[i])

# %%
from models.nn_mlpc import create_mlpc
from utils import compile_and_train_early_strop, visualize_history

batch_size = 32
epochs = 200
num_classes = 28
model_input_iner = Input(shape=(107, 6))
model_iner = create_mlpc(model_input_iner, num_classes)

model_input_ske = Input(shape=(41, 60))
model_ske = create_mlpc(model_input_ske, num_classes)

avg_val_acc_ske = 0
avg_loss_ske = 0
avg_val_acc_iner = 0
avg_loss_iner = 0

for i in range(5):
    X_train_iner = trainX_inertial[i]
    y_train_iner = trainY_inertial[i]
    X_test_iner = testX_inertial[i]
    y_test_iner = testY_inertial[i]

    X_train_ske = trainX_skeleton[i]
    y_train_ske = trainY_skeleton[i]
    X_test_ske = testX_skeleton[i]
    y_test_ske = testY_skeleton[i]

    hist_iner = compile_and_train_early_strop(model_iner, X_train_iner, y_train_iner, X_test_iner, y_test_iner,
                                              batch_size, num_epochs=epochs)

    hist_ske = compile_and_train_early_strop(model_ske, X_train_ske, y_train_ske, X_test_ske, y_test_ske,
                                             batch_size, num_epochs=epochs)

    visualize_history(hist_iner, 'inertial_%d-' % i)
    visualize_history(hist_ske, 'skeleton_%d-' % i)

    avg_loss_ske += hist_ske.history['val_loss'][-1]
    avg_val_acc_ske += hist_ske.history['val_acc'][-1]

    avg_loss_iner += hist_iner.history['val_loss'][-1]
    avg_val_acc_iner += hist_iner.history['val_acc'][-1]

print("ske average loss: " + str(avg_loss_ske / 5))
print("ske average accuracy: " + str(avg_val_acc_ske / 5))
print("iner average loss: " + str(avg_loss_iner / 5))
print("iner average accuracy: " + str(avg_val_acc_iner / 5))
