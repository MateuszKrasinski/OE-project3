from crossover import Crossover
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection


class AlgorithmParams:
    def __init__(self, grade_strategy, selection, crossover,mutation, size_population, probability_mutation,
                 probability_crossover, number_iteration):
        self.grade_strategy = grade_strategy
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.size_population = size_population
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        self.number_iteration = number_iteration

    def operators_results(self):
        return f"Min/max: {self.grade_strategy}, Selection: {self.selection}, Crossover: {self.crossover}, \n" \
               f"Mutation: {self.mutation}, Population: {self.size_population}, Probability mutation: \n" \
               f"{self.probability_mutation}, Probability crossover: {self.probability_crossover}, " \
               f"Number of epochs: {self.number_iteration} "

    def file_path(self):
        return f"Grade_strategy_{self.grade_strategy}_Selection_{self.selection}_Crossover_{self.crossover}_" \
               f"Mutation_{self.mutation}"
