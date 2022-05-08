import random
from deap import tools


class Crossover:
    one_point = "one_point"
    two_point = "two_point"
    uniform = "uniform"

    allCrossover = [one_point, two_point]

    def __init__(self, name, toolbox):
        self.name = name
        if name == self.one_point:
            self._one_point(toolbox)
        elif name == self.two_point:
            self._two_point(toolbox)
        elif name == self.uniform:
            self._uniform(toolbox)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Crossover.allCrossover)):
            result += f"{i} - {Crossover.allCrossover[i]}, "
        result += "} = "
        return result

    @staticmethod
    def _one_point(toolbox):
        toolbox.register("mate", tools.cxOnePoint)

    @staticmethod
    def _two_point(toolbox):
        toolbox.register("mate", tools.cxTwoPoint)

    @staticmethod
    def _uniform(toolbox):
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
