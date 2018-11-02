#############################################
#                                           #
#       Flow of the training process        #
#                                           #
#############################################
from GA_LSTM.genetic import GeneticEngine
from datetime import datetime as dt
from GA_LSTM import dataPreparation as cd

start = dt.now()
X_train, X_test, Y_train ,Y_test = cd.definitions()


print("::::: [main] Load data ", dt.now() - start, ":::::")

start = dt.now()
engine = GeneticEngine(X_train, Y_train,X_test, Y_test)
print("::::: [main] Initialize GeneticEngine ", dt.now() - start, ":::::")

start = dt.now()
best_ind = engine.best_ind()
print("::::: [main] Find the best individual ", dt.now() - start, ":::::")

