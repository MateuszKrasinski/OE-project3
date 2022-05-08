from deap import tools


class Selection:
    tournament = "tournament"
    best = "best"
    random = "random"
    worst = "worst"
    roulette = "roulette"

    allSelection = [best, tournament, random, worst, roulette]

    def __init__(self, name, toolbox):
        self.name = name
        if name == Selection.best:
            Selection._best(toolbox)
        elif name == Selection.tournament:
            Selection._tournament(toolbox)
        elif name == Selection.random:
            Selection._random(toolbox)
        elif name == Selection.worst:
            Selection._worst(toolbox)
        elif name == Selection.roulette:
            Selection._roulette(toolbox)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Selection.allSelection)):
            result += f"{i} - {Selection.allSelection[i]}, "
        result += "} = "
        return result

    @staticmethod
    def _tournament(toolbox):
        toolbox.register("select", tools.selTournament, tournsize=3)

    @staticmethod
    def _random(toolbox):
        toolbox.register("select", tools.selRandom)

    @staticmethod
    def _best(toolbox):
        toolbox.register("select", tools.selBest)

    @staticmethod
    def _worst(toolbox):
        toolbox.register("select", tools.selWorst)

    @staticmethod
    def _roulette(toolbox):
        toolbox.register("select", tools.selRoulette)
