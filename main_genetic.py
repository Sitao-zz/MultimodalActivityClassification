#############################################
#                                           #
#       Flow of the training process        #
#                                           #
#############################################
from genetic_lstm.genetic import GeneticEngine
from datetime import datetime as dt
import common.dataprep_iner as cd

start = dt.now()
X_train, X_test, Y_train, Y_test, numClass = cd.definitions()

print("::::: [main] Load data ", dt.now() - start, ":::::")

start = dt.now()
engine = GeneticEngine(X_train, Y_train, X_test, Y_test, numClass)
print("::::: [main] Initialize GeneticEngine ", dt.now() - start, ":::::")

start = dt.now()
best_ind = engine.best_ind()
print("::::: [main] Find the best individual ", best_ind, dt.now() - start, ":::::")
