from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length, features, saved_model=None):
        """
        `model` = one of:
            lstm
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        #metrics = ['accuracy']        
        metrics = ['accuracy']
        
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features)
            self.model = self.lstm()
        elif model == 'lstm_hof':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features)
            self.model = self.lstm_hof()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-4, decay=1e-6, clipnorm=0.6)
        #optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.00001)
        #optimizer = SGD(lr=0.2, sdecay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
        #self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
        #                   metrics=metrics)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())
    
    def lstm_hof(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predominently."""
        # Model.
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=self.input_shape,
                       dropout=0.3, recurrent_dropout=0.3))
        model.add(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
    
    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predominently."""
        # Model.
        model = Sequential()
        model.add(LSTM(512, return_sequences=True, input_shape=self.input_shape,
                       dropout=0.3, recurrent_dropout=0.3))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model