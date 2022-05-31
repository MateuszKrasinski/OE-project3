import math
import multiprocessing
import random

from deap import base
from deap import creator
from deap import tools

from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from crossover import Crossover
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection
from utils import draw_chart, save_results_to_csv
from numpy.random import randint
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
df=pd.read_csv("C:/Users/milos/Downloads/oe4.csv", sep=',')
y=df['Cath']
df.drop('Cath', axis=1, inplace=True)
# df.drop('ID', axis=1, inplace=True)
# df.drop('Recording', axis=1, inplace=True)
numberOfAtributtes= len(df.columns)
print(numberOfAtributtes)

mms = MinMaxScaler()
df_norm = mms.fit_transform(df)
clf = SVC()
scores = model_selection.cross_val_score(clf, df_norm, y,
                                         cv=5, scoring='accuracy',n_jobs=-1)
print(scores.mean())



class GeneticAlgorithm:
    @staticmethod
    def SVCParameters(numberFeatures, icls):
        genome = list()
        # kernel
        # listKernel = ["scale", " auto"]
        # genome.append(listKernel[random.randint(0, 1)])
        # c
        k = random.uniform(0.1, 100)
        genome.append(k)
        # degree
        genome.append(random.randint(1, 5))
        # gamma
        gamma = random.uniform(0.001, 5)
        genome.append(gamma)
        # coeff
        coeff = random.uniform(0.01, 10)
        genome.append(coeff)
        return icls(genome)

    @staticmethod
    def SVCParametersFitness(y, df, numberOfAtributtes, individual):
        split = 5
        cv = StratifiedKFold(n_splits=split)
        mms = MinMaxScaler()
        df_norm = mms.fit_transform(df)
        estimator = SVC(C=individual[0], degree=individual[1],
                        gamma=individual[2], coef0 = individual[3], random_state = 101)
        resultSum = 0
        for train, test in cv.split(df_norm, y):
            if type(df_norm[train]) == str or type(y[train]) == str:
                break
            estimator.fit(df_norm[train], y[train])
            predicted = estimator.predict(df_norm[test])
            expected = y[test]
            tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
            result = (tp + tn) / (tp + fp + tn + fn)

            resultSum = resultSum + result  #
        return resultSum / split,

    def mutationSVC(individual):
        numberParamer = random.randint(0, len(individual) - 1)
        if numberParamer == 0:
            # kernel
            # listKernel = ["scale", " auto"]
            # individual[0] = listKernel[random.randint(0, 1)]
            pass
        elif numberParamer == 1:
            k = random.uniform(0.1, 100)
            individual[0] = k
        elif numberParamer == 2:
            # degree
            individual[1] = random.uniform(0.1, 5)
        elif numberParamer == 3:
            # gamma
            gamma = random.uniform(0.01, 5)
            individual[2] = gamma
        elif numberParamer == 4:
            # coeff
            coeff = random.uniform(0.1, 20)
            individual[3] = coeff

    @staticmethod
    def decodeInd(individual):
        chromosome_length = 20
        bounds = [[-1.5, 4.0], [-3.0, 4.0]]
        decoded = list()
        largest = 2 ** chromosome_length
        for i in range(len(bounds)):
            # extract the substring
            start, end = i * \
                         chromosome_length, (i * chromosome_length) + \
                         chromosome_length
            substring = individual[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = bounds[i][0] + (integer / largest) * \
                    (bounds[i][1] - bounds[i][0])
            # store
            decoded.append(value)
        return decoded

    @staticmethod
    def individual(icls):
        genome = list()
        for x in range(0, 40):
            genome.append(randint(0, 2))
        return icls(genome)

    @staticmethod
    def fitness_function(individual):
        ind = GeneticAlgorithm.decodeInd(individual)
        result = math.sin((ind[0] + ind[1])) + math.pow(
            (ind[0] - ind[1]), 2) - 1.5 * ind[0] + 2.5 * ind[1] + 1
        return result,

    @staticmethod
    def pass_operators(algorithm_params):
        algorithm_params.grade_strategy = GradeStrategy.grade_strategies[int(input(GradeStrategy.options()))]
        algorithm_params.selection = Selection.allSelection[int(input(Selection.options()))]
        algorithm_params.crossover = Crossover.allCrossover[int(input(Crossover.options()))]
        algorithm_params.mutation = Mutation.allMutation[int(input(Mutation.options()))]

    @staticmethod
    def register_operators(toolbox, algorithm_params):
        GradeStrategy(algorithm_params.grade_strategy)
        Selection(algorithm_params.selection, toolbox)
        Crossover(algorithm_params.crossover, toolbox)
        Mutation(algorithm_params.mutation, toolbox)

    @staticmethod
    def register_functions(toolbox):
        toolbox.register("individual", GeneticAlgorithm.SVCParameters, numberOfAtributtes, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", GeneticAlgorithm.SVCParametersFitness, y, df, numberOfAtributtes)
        # toolbox.register("individual", GeneticAlgorithm.individual, creator.Individual)
        # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # toolbox.register("evaluate", GeneticAlgorithm.fitness_function)

    @staticmethod
    def run(algorithm_params, processes=1):
        global best_ind, mean, std, invalid_ind
        toolbox = base.Toolbox()
        GeneticAlgorithm.register_operators(toolbox, algorithm_params)
        GeneticAlgorithm.register_functions(toolbox)

        pop = toolbox.population(n=algorithm_params.size_population)
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            tmp = [fit]
            ind.fitness.values = fit

        g, number_elitism = 0, 1
        best_results, avg_results, std_results = [], [], []
        while g < algorithm_params.number_iteration:
            g = g + 1

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(toolbox.map(toolbox.clone, offspring))

            list_elitism = []
            for x in range(0, number_elitism):
                list_elitism.append(tools.selBest(pop, 1)[0])

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() < algorithm_params.probability_crossover:
                    toolbox.mate(child1, child2)
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < algorithm_params.probability_mutation:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            if __name__ == "__main__":
                pool = multiprocessing.Pool(processes=processes)
                toolbox.register("map", pool.map)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring + list_elitism

            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = tools.selBest(pop, 1)[0]

            best_results.append(best_ind.fitness.values)
            avg_results.append(mean)
            std_results.append(std)

        save_results_to_csv(best_ind, best_ind, mean, std, algorithm_params)
        draw_chart(algorithm_params, best_results, avg_results, std_results, g)
