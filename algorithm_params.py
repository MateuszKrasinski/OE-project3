from classifier import Classifiers
from crossover import Crossover
from grade_strategy import GradeStrategy
from mutation import Mutation
from selection import Selection


class AlgorithmParams:
    def __init__(self, grade_strategy=GradeStrategy.min, selection=Selection.best, crossover=Crossover.one_point,
                 mutation=Mutation.gaussian, size_population=20, probability_mutation=0.2, probability_crossover=0.8,
                 number_iteration=20, with_deap=True, classifier=Classifiers.own):
        self.grade_strategy = grade_strategy
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.size_population = size_population
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        self.number_iteration = number_iteration
        self.with_deap = with_deap
        self.classifier = classifier

    def operators_results(self):
        return f"Classifier: {self.classifier}, Grade strategy: {self.grade_strategy}, Selection: {self.selection}, " \
               f"Crossover: {self.crossover}, " \
               f"Mutation: {self.mutation}, Size population: {self.size_population}, Probability mutation: " \
               f"{self.probability_mutation}, Probability crossover: {self.probability_crossover}, " \
               f"Number iteration: {self.number_iteration} "

    def file_path(self):
        return f"Classifier_{self.classifier}_Grade_strategy_{self.grade_strategy}_Selection_{self.selection}_Crossover_{self.crossover}_" \
               f"Mutation_{self.mutation}"
