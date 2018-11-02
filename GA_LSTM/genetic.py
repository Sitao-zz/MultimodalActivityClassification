from GA_LSTM.model import Model
import random
import numpy

from deap import tools
from deap import creator
from deap import base
from deap import algorithms
from datetime import datetime as dt

IND_INIT_SIZE = 10
NBR_ITEMS = 2
NN_MIN, NN_MAX = 100, 600
Dp_MIN,Dp_MAX = 0.01, 0.5
N_CYCLES = 1


class GeneticEngine:

    def __init__(self, X_train, Y_train, X_test, Y_test,numClass):
        self.model = Model(X_train, Y_train, X_test, Y_test,numClass)

        # To assure reproductibility, the RNG seed is set prior to the items
        # dict initialization. It is also seeded in main().
        random.seed(64)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_nn", random.randint, NN_MIN, NN_MAX)
        #self.toolbox.register("attr_ne", random.randint, NE_MIN, NE_MAX)
        self.toolbox.register("attr_dp",random.uniform,Dp_MIN,Dp_MAX)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.attr_nn, self.toolbox.attr_dp), n=N_CYCLES)

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_ind)
        self.toolbox.register("mate", self.cx_ind)
        self.toolbox.register("mutate", self.mutate_ind)
        self.toolbox.register("select", tools.selNSGA2)

    @property
    def evaluator(self):
        return self._evaluator

    def eval_ind(self, ind):
        """
        calculate the fitness of the individual

        :param ind: the individual Chromosome object to be evaluated
        :return: the fitness value
        """
        print("\n:::: [genetic] individual", ind, "::::")
        start = dt.now()
        # {260, 5}
        num_nuros = list(ind)[0]
        num_dp = list(ind)[1]
        #num_dp=list(ind)[2]

        print("number of neuros :", num_nuros)
       # print("number of num_epoch :", num_epoch)
        fit_val = self.model.evaluate(num_nuros, num_dp)
        print(":::: [genetic] Evaluate individual. fitness value", fit_val, "Duration", dt.now() - start, "::::\n")
        print(ind)
        return fit_val, None

    def mutate_ind(self, ind, mu=0, sigma=4, chance_mutation=0.4):
        """
        Mutate the individual by changing the Chromosome composition
        :param mu:
        :param sigma:
        :return:
        """
        if random.random() < chance_mutation:
            if len(ind) > 0:  # We cannot pop from an empty set
                ind.remove(random.choice(sorted(tuple(ind))))
        else:
            ind.add(random.randrange(NBR_ITEMS))
        return ind

    def cx_ind(self, ind1, ind2, chance_crossover=0.7):
        """Apply a crossover operation on input sets. The first child is the
        intersection of the two sets, the second child is the difference of the
        two sets.
        """
        if random.random() < chance_crossover:
            temp = set(ind1)  # Used in order to keep type
            ind1 &= ind2  # Intersection (inplace)
            ind2 ^= temp  # Symmetric Difference (inplace)
        return ind1, ind2

    def verify_ind(self, ind):
        """
        Verify the validity of the individual Chromosome

        :param ind: the individual Chromosome to be verified
        :return: True if the Chromosome is valid, False otherwise
        """
        # TODO: will it be better if we define the method as the member method of Chromosome?
        return True

    def best_ind(self):
        """
        Get the best individual after the GA calculation

        :return: the best individual Chromosome
        """
        random.seed(64)
        NGEN = 10
        MU = 10
        pop = self.toolbox.population(n=MU)
        hof = tools.ParetoFront()
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)
        #algorithms.eaMuPlusLambda(pop, self.toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
        algorithms.eaSimple(pop, self.toolbox, cxpb=0.6, mutpb=0.05, ngen=NGEN, halloffame=hof)
        print("The best individual is :", hof[-1])
        # print("The best fitness is :", eval_ind(self, hof[-1]))
        return hof[-1]
