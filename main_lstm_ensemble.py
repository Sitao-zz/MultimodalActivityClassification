# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:14:10 2017

@author: HP
"""

import os

import numpy as np
import pandas as pd
import scipy.io as scio

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
from models.lstm_ensemble import create_lstm_ensemble
model = create_lstm_ensemble()

# %%
from keras.callbacks import EarlyStopping

avg_val_acc = 0
avg_mae = 0
avg_loss = 0
avg_val_acc_ske = 0
avg_mae_ske = 0
avg_loss_ske = 0
avg_val_acc_iner = 0
avg_mae_iner = 0
avg_loss_iner = 0

for i in range(5):
    X_iner = trainX_inertial[i]
    X_ske = trainX_skeleton[i]
    hist = model.fit([X_iner, X_ske], [trainY_skeleton[i], trainY_inertial[i], trainY_skeleton[i]], validation_data=(
        [testX_inertial[i], testX_skeleton[i]], [testY_skeleton[i], testY_skeleton[i], testY_skeleton[i]]),
                     callbacks=[EarlyStopping(monitor='val_main_output_acc', patience=10, verbose=1, mode='auto')],
                     epochs=200)
    avg_mae += hist.history['val_main_output_mean_absolute_error'][-1]
    avg_loss += hist.history['val_main_output_loss'][-1]
    avg_val_acc += hist.history['val_main_output_acc'][-1]
    avg_mae_ske += hist.history['val_skeleton_output_mean_absolute_error'][-1]
    avg_loss_ske += hist.history['val_skeleton_output_loss'][-1]
    avg_val_acc_ske += hist.history['val_skeleton_output_acc'][-1]
    avg_mae_iner += hist.history['val_inertial_output_mean_absolute_error'][-1]
    avg_loss_iner += hist.history['val_inertial_output_loss'][-1]
    avg_val_acc_iner += hist.history['val_inertial_output_acc'][-1]

print("average mae : " + str(avg_mae / 5))
print("average loss : " + str(avg_loss / 5))
print("average accuracy: " + str(avg_val_acc / 5))
print("ske average mae: " + str(avg_mae_ske / 5))
print("ske average loss: " + str(avg_loss_ske / 5))
print("ske average accuracy: " + str(avg_val_acc_ske / 5))
print("iner average mae: " + str(avg_mae_iner / 5))
print("iner average loss: " + str(avg_loss_iner / 5))
print("iner average accuracy: " + str(avg_val_acc_iner / 5))
