from deap import creator

from fitness import Fitnesses
from individual import Individual
from deap import tools


class Classifiers(object):
    own = "own"
    svc = "svc"
    decision_tree_classifier = "decision_tree_classifier"
    k_neighbors_classifier = "k_neighbors_classifier"
    extra_tree_classifier = "extra_tree_classifier"
    mlp_classifier = "mlp_classifier"
    random_forrest_classifier = "random_forrest_classifier"
    allClassifiers = [own, svc, decision_tree_classifier, k_neighbors_classifier, extra_tree_classifier, mlp_classifier]

    @staticmethod
    def register(name, toolbox, y=None, df=None, number_of_attributes=None):
        if name == Classifiers.own:
            toolbox.register("individual", Individual.individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.fitness_function)
        elif name == Classifiers.svc:
            toolbox.register("individual", Individual.svc, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.svc, y, df, number_of_attributes)
        elif name == Classifiers.decision_tree_classifier:
            toolbox.register("individual", Individual.decision_tree_classifier, number_of_attributes,
                             creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.decision_tree_classifier, y, df, number_of_attributes)
        elif name == Classifiers.k_neighbors_classifier:
            toolbox.register("individual", Individual.k_neighbors_classifier, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.k_neighbors_classifier, y, df, number_of_attributes)
        elif name == Classifiers.extra_tree_classifier:
            toolbox.register("individual", Individual.extra_tree_classifier, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.extra_tree_classifier, y, df, number_of_attributes)
        elif name == Classifiers.mlp_classifier:
            toolbox.register("individual", Individual.mlp_classifier, number_of_attributes, creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.mlp_classifier, y, df, number_of_attributes)
        elif name == Classifiers.random_forrest_classifier:
            toolbox.register("individual", Individual.random_forrest_classifier, number_of_attributes,
                             creator.Individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", Fitnesses.random_forrest_classifier, y, df, number_of_attributes)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Classifiers.allClassifiers)):
            result += f"{i} - {Classifiers.allClassifiers[i]}, "
        result += "} = "
        return result
