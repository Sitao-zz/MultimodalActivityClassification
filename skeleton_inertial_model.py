from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import np_utils
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from dataprep import definitions
import pandas as pd
import numpy as np
import pickle as pk
import time
from modeldef import ResearchModels
import tensorflow as tf
from keras import backend as K

def train(data, model, seq_length, feature_selection, features, classes, 
          saved_model=None, concat=False):
    # Set variables.
    nb_epoch = 300
    batch_size = 32

    f1_results, precision_results, recall_results = [], [], []

    for k in range(5):
        X, y, X_val, y_val = [], [], [], []
        for row in data.itertuples(): 
            sequence = []
            for f in range(maxlength):
                v = []
                for col in range(len(row[1:-1])):
                    v.append(row[1:-1][col][f])
                sequence.append(v)
            if concat:
                # We want to pass the sequence back as a single array. This
                # is used to pass into a CNN or MLP, rather than an RNN.
                sequence = np.concatenate(sequence).ravel()
            yencode = np_utils.to_categorical(classes.index(str(row.target)), len(classes))[0] # One-hot encoding
            if row.Index in trainsets[k]:
                X.append(sequence)
                y.append(yencode)
            elif row.Index in validationsets[k]:
                X_val.append(sequence)
                y_val.append(yencode)
        X = np.array(X)
        y = np.array(y)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        # Get the model.
        rm = ResearchModels(len(classes), model, seq_length, features, saved_model)
        
        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath='./data/checkpoints/' + model + feature_selection + str(k) + '.hdf5',
            verbose=1,
            save_best_only=True)
    
        # Helper: TensorBoard
        tb = TensorBoard(log_dir='./data/logs')
    
        # Helper: Stop when we stop learning.
        early_stopper = EarlyStopping(patience=50)
    
        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger('./data/logs/' + model + feature_selection + str(k) + '-training-' + \
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
        actual = [ int(i) for i, j in zip(data.target, data.index) if j in validationsets[k] ]
        pred = [ int(classes[int(np.argmax(i))]) for i in results ]
        
        # Save predictions to csv for ensemble scoring later
        prediction_output = pd.DataFrame()
        prediction_output['Predictions'] = pred
        prediction_output.to_csv('./results/predictions'+feature_selection+str(k)+'.csv')
        
        report = classification_report(actual, pred, classes, digits=2)
        f1 = f1_score(actual ,pred, classes, average='macro')
        precision = precision_score(actual, pred, classes, average='macro')
        recall = recall_score(actual, pred, classes, average='macro')
        f1_results.append(f1)
        precision_results.append(precision)
        recall_results.append(recall)
        print(report)
        print(f1)
        text_file = open("./results/report"+feature_selection+str(k)+".txt", "w")
        
        text_file.write(report)
        text_file.close()
        
        K.clear_session()
    
    avgf1 = np.mean(np.array(f1_results))
    avgprecision = np.mean(np.array(precision_results))
    avgrecall = np.mean(np.array(recall_results))
    print('Average precision score: {0:0.3f}'.format(avgprecision))
    print('Average recall score: {0:0.3f}'.format(avgrecall))
    print('Average F1 score: {0:0.3f}'.format(avgf1))
    
if __name__ == '__main__':
    
    dataset, trainsets, validationsets = definitions()
    
    # joint data
    distance_list = ['hand_d','foot_d','hand_foot_l_d','hand_foot_r_d',
                     'head_hand_l_d','head_hand_r_d', 'elbow_l_a', 'elbow_r_a']
    pos_list = ['hand_l_x','hand_r_x','hand_l_y','hand_r_y','hand_l_z','hand_r_z',
           'foot_l_x','foot_r_x','foot_l_y','foot_r_y','foot_l_z','foot_r_z',
           'head_x','head_y','head_z','hip_center_x','hip_center_y','hip_center_z']
    inertial_list = ['acc_x','acc_y','acc_z','rot_x','rot_y','rot_z',]
    # Load dataset
    all_data = pk.load(open("./all_df.pk", "rb"))
    
    maxlength = len(all_data.head_x[0])
    classes = []
    for item in all_data.itertuples():
        if item[-1] not in classes:
            classes.append(item[-1])
    classes = sorted(classes)
    
    """These are the main training settings. Set each before running
    this file."""
    model = 'lstm'  # see `modeldef.py` for more
    saved_model = None  # None or weights file
    seq_length = maxlength
    feature_selection = 'distances_joints_inertial'
    
    if feature_selection == 'all':
        data = all_data
    elif feature_selection == 'distances_joints':
        data = all_data[distance_list+pos_list+['target']]
    elif feature_selection == 'joints_inertial':
        data = all_data.drop(distance_list, axis=1)
    elif feature_selection == 'distances':
        data = all_data[distance_list+['target']]
    elif feature_selection == 'distances_joints_inertial':
        data = all_data[distance_list+pos_list+inertial_list+['target']]
    elif feature_selection == 'distances_inertial':
        data = all_data[distance_list+inertial_list+['target']]
    elif feature_selection == 'inertial':
        data = all_data[inertial_list+['target']]
    
    features = len(data.columns)-1

    # MLP requires flattened features.
    if model == 'mlp':
        concat = True
    else:
        concat = False
    
    train(data, model, seq_length, feature_selection, features, classes, saved_model=saved_model, concat=concat)