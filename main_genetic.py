#############################################
#                                           #
#       Flow of the training process        #
#                                           #
#############################################
from datetime import datetime as dt
from genetic_lstm.genetic import GeneticEngine
from common.dataprep_iner import definitions

start = dt.now()
X_train, X_test, Y_train, Y_test, num_classes = definitions()

print("::::: [main] Load data ", dt.now() - start, ":::::")

start = dt.now()
engine = GeneticEngine(X_train, Y_train, X_test, Y_test, num_classes)
print("::::: [main] Initialize GeneticEngine ", dt.now() - start, ":::::")

start = dt.now()
best_ind = engine.best_ind()
print("::::: [main] Find the best individual ", best_ind, dt.now() - start, ":::::")
