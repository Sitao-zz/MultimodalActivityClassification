from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from keras.preprocessing import sequence
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from models.lstm_simple import create_lstm_simple


def import_depth_data(action, subject, trial):
    filename = f'dataset/Depth/a{action}_s{subject}_t{trial}_depth.mat'
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_depth']
    else:
        return None


def transform_depth_data(action, subject, trial):
    rows = []
    data = import_depth_data(action, subject, trial)
    if data is None: return None
    for frame in range(data.shape[2]):
        pixels = data[:, :, frame].flatten()
        rows.append(pixels)
    result = np.insert(rows, 0, [[action], [subject], [trial], [frame]], axis=1)
    return np.array(result)


def transform_depth_data_to_df(action, subject, trial):
    data = transform_depth_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df.columns = ['action', 'subject', 'trial', 'frame'] + [f'depth_{n}' for n in range(240 * 320)]
    return df


def export_depth_data_to_csv(action, subject, trial):
    df = transform_depth_data_to_df(action, subject, trial)
    if df is None: return None
    filename = f'a{action}_s{subject}_t{trial}_depth.csv'
    df.to_csv(filename, index=False)


def show_depth_image(action, subject, trial, frame):
    data = import_depth_data(action, subject, trial)
    if data is None: return None
    plt.imshow(data[:, :, frame], cmap='gray')
    plt.axis('off')
    plt.show()


def import_inertial_data(action, subject, trial):
    filename = f'dataset/Inertial/a{action}_s{subject}_t{trial}_inertial.mat'
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_iner']
    else:
        return None


def transform_inertial_data(action, subject, trial):
    data = import_inertial_data(action, subject, trial)
    if data is None: return None
    result = np.insert(data, 0, [[action], [subject], [trial]], axis=1)
    return np.array(result)


def transform_inertial_data_to_df(action, subject, trial):
    data = transform_inertial_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df.columns = ['action', 'subject', 'trial', 'x-accel', 'y-accel', 'z-accel', 'x-ang-accel', 'y-ang-accel',
                  'z-ang-accel']
    return df


def export_inertial_data_to_csv(action, subject, trial):
    df = transform_inertial_data_to_df(action, subject, trial)
    if df is None: return None
    filename = f'a{action}_s{subject}_t{trial}_inertial.csv'
    df.to_csv(filename, index=False)


def import_skeleton_data(action, subject, trial):
    filename = f'dataset/Skeleton/a{action}_s{subject}_t{trial}_skeleton.mat'
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_skel']
    else:
        return None


def transform_skeleton_data(action, subject, trial):
    matrices = []
    data = import_skeleton_data(action, subject, trial)
    if data is None: return None
    for frame in range(data.shape[2]):
        skelecton_joints = [i + 1 for i in range(20)]
        matrix = data[:, :, frame]
        matrix = np.insert(matrix, 0, skelecton_joints, axis=1)
        matrix = np.insert(matrix, 0, frame, axis=1)
        matrices.append(matrix)
    result = np.vstack(tuple(matrices))
    result = np.insert(result, 0, [[action], [subject], [trial]], axis=1)
    return result


def transform_skeleton_data_to_df(action, subject, trial):
    data = transform_skeleton_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df.columns = ['action', 'subject', 'trial', 'frame', 'skeleton_joint', 'x', 'y', 'z']
    return df


def export_inertial_data_to_csv(action, subject, trial):
    df = transform_skeleton_data_to_df(action, subject, trial)
    if df is None: return None
    filename = f'a{action}_s{subject}_t{trial}_skeleton.csv'
    df.to_csv(filename, index=False)


df = transform_depth_data_to_df(1, 1, 1)
df.head()

df = transform_inertial_data_to_df(1, 1, 1)
df.head()

df = transform_skeleton_data_to_df(1, 1, 1)
df.head()

show_depth_image(1, 1, 1, 1)

# Original inertial data has dimension (Number of sample) x 6
# Swap the axes so the new dimension is 6 x (Number of sample)
# Apply padding to each entry, the new dimension is 6 x 326
# Subjects 1, 2, 3, 5, 6, 7 go into training data (75%)
# Subjects 4, 8 go into test data (25%)

X_train = []
Y_train = []
X_test = []
Y_test = []

# Select 3 activites for training:
#     3. right hand wave (wave)
#     18. two hand push (push)
#     22. jogging (jog)
activities = [3, 18, 22]

for index, action in enumerate(activities):
    for subject in range(1, 9):
        for trial in range(1, 5):
            data = import_inertial_data(action, subject, trial)
            if data is None: continue
            data = np.swapaxes(data, 0, 1)
            data = sequence.pad_sequences(data, maxlen=326)
            if subject in [1, 2, 3, 5, 6, 7]:
                X_train.append(data)
                Y_train.append(index)
            else:
                X_test.append(data)
                Y_test.append(index)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

# Swap axes again, new dimension is 32 x 326 x 6
# This follows the standard of LSTM: Samples, Timesteps, Dimensions
X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)

print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

# One hot encoding
Y_train = to_categorical(Y_train, num_classes=len(activities))
Y_test = to_categorical(Y_test, num_classes=len(activities))

print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_test.shape:', X_test.shape)
print('Y_test.shape:', Y_test.shape)

from keras.callbacks import EarlyStopping

model = create_lstm_simple(len(activities))

# Train model
history = model.fit(X_train, Y_train, callbacks=[EarlyStopping(monitor='acc', patience=10, verbose=1, mode='auto')],
                    epochs=10, batch_size=1)

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
