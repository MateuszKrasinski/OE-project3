from algorithm_params import AlgorithmParams
from crossover import Crossover
from genetic_algorithm import GeneticAlgorithm
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection
from utils import add_headers_csv


def run():
    add_headers_csv()
    for gs in GradeStrategy.grade_strategies:
        for sel in Selection.allSelection:
            for cx in Crossover.allCrossover:
                for mut in Mutation.allMutation:
                    print(f"{gs} {sel} {cx} {mut}")
                    GeneticAlgorithm.run(
                        AlgorithmParams(gs, sel, cx, mut, size_population=100, probability_mutation=0.2,
                                        probability_crossover=0.8, number_iteration=200), False)


if __name__ == '__main__':
    run()
