import time

from algorithm_params import AlgorithmParams
from classifier import Classifiers
from crossover import Crossover
from genetic_algorithm import GeneticAlgorithm
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection


def run():
    GeneticAlgorithm.run(
        AlgorithmParams(GradeStrategy.max, Selection.best, Crossover.one_point, Mutation.shuffle_indexes,
                        size_population=100, probability_mutation=0.2, probability_crossover=0.8,
                        number_iteration=100, classifier=Classifiers.extra_tree_classifier), processes=8,
        use_global_operators=False)


if __name__ == '__main__':
    run()
