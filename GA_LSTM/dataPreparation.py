import scipy.io as sio
import pandas as pd
import scipy.io
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from pathlib import Path
from keras.utils import to_categorical
import matplotlib
matplotlib.use('AGG')
import numpy as np


from sklearn.model_selection import train_test_split

def definitions():

    def import_depth_data(action, subject, trial):
        filename = f'../dataset/Depth/a{action}_s{subject}_t{trial}_depth.mat'
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

    def import_inertial_data(action, subject, trial):
        filename = f'../dataset/Inertial/a{action}_s{subject}_t{trial}_inertial.mat'
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

    df = transform_depth_data_to_df(1, 1, 1)
    df.head()

    df = transform_inertial_data_to_df(1, 1, 1)
    df.head()

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
    return X_train, X_test, Y_train ,Y_test
