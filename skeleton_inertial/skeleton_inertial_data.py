import scipy.io as sio
import pandas as pd
import numpy as np
import pickle as pk
from common.dataprep import definitions

dataset, trainsets, validationsets = definitions()

""" Inertial Dataset """
inertial_dataset = {i: {'inertial': sio.loadmat('dataset/Inertial/' + i + '_inertial.mat')['d_iner']} for i in dataset}

""" Skeleton Dataset """
skeletons_dataset = {i: {'sk': sio.loadmat('dataset/Skeleton/' + i + '_skeleton.mat')['d_skel']} for i in dataset}

jointType = {'hip_center': 0, 'spine': 1, 'shoulder_c': 2, 'head': 3,
             'shoulder_r': 4, 'elbow_r': 5, 'wrist_r': 6, 'hand_r': 7,
             'shoulder_l': 8, 'elbow_l': 9, 'wrist_l': 10, 'hand_l': 11,
             'hip_r': 12, 'knee_r': 13, 'ankle_r': 14, 'foot_r': 15, 'hip_l': 16,
             'knee_l': 17, 'ankle_l': 18, 'foot_l': 19}

distances = ['hand_d', 'foot_d', 'hand_foot_l_d', 'hand_foot_r_d',
             'head_hand_l_d', 'head_hand_r_d', 'elbow_l_a', 'elbow_r_a']

inertials = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', ]

columns = []
for i in jointType.keys():
    for j in ['x', 'y', 'z']:
        columns.append(i + '_' + j)

columns = columns + distances + inertials


# Joint Smoothing
# ['a1_s1_t1']['sk'][20][3][sequence-length]

def calEuclidean(a, b):
    # Input: ndarrays of joint a and joint b
    ax, bx, ay, by, az, bz = a[0], b[0], a[1], b[1], a[2], b[2]
    output = []
    for i in range(len(ax)):
        output.append(np.linalg.norm(np.array([ax[i], ay[i], az[i]]) - np.array([bx[i], by[i], bz[i]])))
    return np.array(output)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(a, b, c):
    ax, bx, cx, ay, by, cy, az, bz, cz = a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]
    angles = []
    for i in range(len(ax)):
        au = np.array([ax[i], ay[i], az[i]])
        bu = np.array([bx[i], by[i], bz[i]])
        cu = np.array([cx[i], cy[i], cz[i]])
        du = au - bu
        angle = np.arccos(np.clip(np.dot(du, cu), -1, 1)) / (np.linalg.norm(du) * np.linalg.norm(cu))
        if angle > (np.pi / 2):
            angles.append((2 * np.pi - angle) / np.pi)
        else:
            angles.append(angle / np.pi)
    return np.array(angles)


# ewma = pd.stats.moments.ewma
for s in dataset:
    print(s)
    # Joint Smoothing
    sk = skeletons_dataset[s]['sk']
    for j in range(len(sk)):
        for c in range(len(sk[j])):
            # take EWMA in both directions with a smaller span term
            coord = sk[j][c]
            fwd = pd.DataFrame(coord).ewm(span=3).mean().values  # take EWMA in forward direction
            bwd = pd.DataFrame(coord[::-1]).ewm(span=3).mean().values  # take EWMA in backward direction
            smoothc = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
            smoothc = np.mean(smoothc, axis=0)  # average
            sk[j][c] = smoothc

    # Intertial Smoothing
    inert = inertial_dataset[s]['inertial']

    for v in range(len(inert[0])):
        # take EWMA in both directions with a smaller span term
        seq = np.array([t[v] for t in inert])
        fwd = pd.DataFrame(seq).ewm(span=10).mean().values  # take EWMA in forward direction
        bwd = pd.DataFrame(seq[::-1]).ewm(span=10).mean().values  # take EWMA in backward direction
        smoothc = np.vstack((fwd, bwd[::-1]))  # lump fwd and bwd together
        smoothc = np.mean(smoothc, axis=0)  # average
        inertial_dataset[s][inertials[v]] = smoothc
        # plt.plot(smoothc, 'b', label='Reversed-Recombined' )

    height = abs(sk[jointType['head']][1][0] - sk[jointType['hip_center']][1][0])

    jointVar = {}
    j = 0
    for i in range(len(jointType)):
        for k in range(3):
            jointVar[columns[j]] = sk[i][k] / height
            j = j + 1

    # Feature Engineering
    keyVar = {}
    keyVar['hand_d'] = calEuclidean(sk[jointType['hand_l']], sk[jointType['hand_r']]) / height
    keyVar['foot_d'] = calEuclidean(sk[jointType['foot_l']], sk[jointType['foot_r']]) / height
    keyVar['hand_foot_l_d'] = calEuclidean(sk[jointType['hand_l']], sk[jointType['foot_l']]) / height
    keyVar['hand_foot_r_d'] = calEuclidean(sk[jointType['hand_r']], sk[jointType['foot_r']]) / height
    keyVar['head_hand_l_d'] = calEuclidean(sk[jointType['head']], sk[jointType['hand_l']]) / height
    keyVar['head_hand_r_d'] = calEuclidean(sk[jointType['head']], sk[jointType['hand_r']]) / height
    keyVar['elbow_l_a'] = angle_between(sk[jointType['shoulder_l']], sk[jointType['elbow_l']],
                                        sk[jointType['wrist_l']])
    keyVar['elbow_r_a'] = angle_between(sk[jointType['shoulder_r']], sk[jointType['elbow_r']],
                                        sk[jointType['wrist_r']])

    for v in jointVar:
        skeletons_dataset[s][v] = jointVar[v]
    for v in keyVar:
        skeletons_dataset[s][v] = keyVar[v]


def rescale(input_list, size):
    skip = max(len(input_list) // size, 1)
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    return output[:size]


# Output Dataframe
all_df = pd.DataFrame(index=dataset, columns=columns)

maxlength = min([len(skeletons_dataset[s]['sk'][0][0]) for s in dataset])
for v in columns:
    if v in inertials:
        maxValue = max([max(abs(inertial_dataset[s][v])) for s in dataset])
        minValue = min([min(inertial_dataset[s][v]) for s in dataset])


        def normalise(v):
            return v / maxValue
            # return (v - maxValue) / (maxValue - minValue)


        all_df[v] = [rescale(normalise(inertial_dataset[s][v]), maxlength) for s in dataset]
    else:
        maxValue = max([max(abs(skeletons_dataset[s][v])) for s in dataset])
        minValue = min([min(skeletons_dataset[s][v]) for s in dataset])


        def normalise(v):
            return v / maxValue
            # return (v - maxValue) / (maxValue - minValue)


        all_df[v] = [rescale(normalise(skeletons_dataset[s][v]), maxlength) for s in dataset]

""" Add Target """

all_df['target'] = [i.split('_')[0][1:] for i in all_df.index]

""" Save Processed Data """

pk.dump(all_df, open("all_df.pk", "wb"))
