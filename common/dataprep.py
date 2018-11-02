import numpy as np
import pandas as pd
import scipy.io as scio
import sklearn.preprocessing


def definitions():
    actionset = ['a' + str(i) for i in range(1, 28)]
    subjectset = ['s' + str(i) for i in range(1, 9)]
    repset = ['t' + str(i) for i in range(1, 5)]

    dataset = ['_'.join([a, s, t]) for s in subjectset for a in actionset for t in repset]
    # Remove missing elements (which are corrupted)
    dataset.remove('a8_s1_t4')
    dataset.remove('a23_s6_t4')
    dataset.remove('a27_s8_t4')

    # trainset = [ d for d in dataset if d.split('_')[1] in 's1s3s5s7']
    # validationset = [  '_'.join([a,s,t]) for s in subjectset for a in actionset for t in repset if s in 's2s4']
    # testset = [ d for d in dataset if d.split('_')[1] in 's2s4s6s8']

    # Implement k-fold cross-validation
    trainingsubjects = [''.join([('s' + str(i + k + 1)) for i in range(4)]) for k in range(5)]
    validationsets, trainsets = [], []
    for i in range(5):
        # validationset = [ s for s in trainset if s.split('_')[1] in ('s'+str(i*2+1)) ]
        # validationsets.append(validationset)
        # trainsets.append([s for s in trainset if s not in validationset])
        trainsets.append([s for s in dataset if s.split('_')[1] in trainingsubjects[i]])
        validationsets.append([s for s in dataset if s.split('_')[1] not in trainingsubjects[i]])

    return dataset, trainsets, validationsets


def pre_processing_inertial(data):  # (200,6)
    predata = pd.DataFrame(data).ewm(span=3).mean().values  # convert the dataframe into numpy array
    if data.shape[0] > 107:
        x = data.shape[0] // 2
        predata = predata[(x - 53):(x + 54), :]

    return predata


def pre_processing_skeleton(data):  # (200,6)
    predata = data
    if data.shape[2] > 41:
        x = data.shape[2] // 2
        predata = predata[:, :, (x - 20):(x + 21)]
    return predata


def get_dataset(trainsets, validationsets):
    trainX_ske = [[], [], [], [], []]
    trainY_ske = [[], [], [], [], []]
    testX_ske = [[], [], [], [], []]
    testY_ske = [[], [], [], [], []]
    trainX_ine = [[], [], [], [], []]
    trainY_ine = [[], [], [], [], []]
    testX_ine = [[], [], [], [], []]
    testY_ine = [[], [], [], [], []]
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
            trainX_ske[i].append(inputdata_skeleton)
            trainY_ske[i].append(int(j.split('_')[0][1:]))

            inputdata_inertial = pre_processing_inertial(data_inertial)
            inputdata_inertial = np.ndarray.tolist(inputdata_inertial)
            trainX_ine[i].append(inputdata_inertial)
            trainY_ine[i].append(int(j.split('_')[0][1:]))

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
            testX_ske[i].append(inputdata_skeleton)
            testY_ske[i].append(int(j.split('_')[0][1:]))

            inputdata_inertial = pre_processing_inertial(data_inertial)
            inputdata_inertial = np.ndarray.tolist(inputdata_inertial)
            testX_ine[i].append(inputdata_inertial)
            testY_ine[i].append(int(j.split('_')[0][1:]))
    # %%
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(28))
    for i in range(5):
        trainX_ske[i] = np.array(trainX_ske[i])
        testX_ske[i] = np.array(testX_ske[i])
        trainY_ske[i] = label_binarizer.transform(trainY_ske[i])
        testY_ske[i] = label_binarizer.transform(testY_ske[i])

        trainX_ine[i] = np.array(trainX_ine[i])
        testX_ine[i] = np.array(testX_ine[i])
        trainY_ine[i] = label_binarizer.transform(trainY_ine[i])
        testY_ine[i] = label_binarizer.transform(testY_ine[i])
        return trainX_ske, trainY_ske, testX_ske, testY_ske, trainX_ine, trainY_ine, testX_ine, testY_ine
