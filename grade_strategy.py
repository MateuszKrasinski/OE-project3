from deap import base, creator


class GradeStrategy:
    min = "min"
    max = "max"

    grade_strategies = [min, max]

    def __init__(self, name):
        self.name = name
        if name == self.min:
            self._min()
        elif name == self.max:
            self._max()
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(GradeStrategy.grade_strategies)):
            result += f"{i} - {GradeStrategy.grade_strategies[i]}, "
        result += "} = "
        return result

    @staticmethod
    def _min():
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    @staticmethod
    def _max():
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
