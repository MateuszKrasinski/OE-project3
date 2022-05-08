import math
import multiprocessing
import random

from deap import base
from deap import creator
from deap import tools

from crossover import Crossover
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection
from utils import draw_chart, save_results_to_csv
from numpy.random import randint


class GeneticAlgorithm:
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
        toolbox.register("individual", GeneticAlgorithm.individual, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", GeneticAlgorithm.fitness_function)

    @staticmethod
    def run(algorithm_params, processes=1):
        global best_ind, mean, std, invalid_ind
        toolbox = base.Toolbox()
        GeneticAlgorithm.register_operators(toolbox, algorithm_params)
        GeneticAlgorithm.register_functions(toolbox)

        pop = toolbox.population(n=algorithm_params.size_population)
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
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

        save_results_to_csv(GeneticAlgorithm.decodeInd(best_ind), best_ind, mean, std, algorithm_params)
        draw_chart(algorithm_params, best_results, avg_results, std_results, g)
