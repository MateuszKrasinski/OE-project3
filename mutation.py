from deap import tools


class Mutation:
    shuffle_indexes = "shuffle_indexes"
    flip_bit = "flip_bit"

    allMutation = [shuffle_indexes, flip_bit]

    def __init__(self, name, toolbox):
        self.name = name
        if name == self.shuffle_indexes:
            self._shuffle_indexes(toolbox)
        elif name == self.flip_bit:
            self._flip_bit(toolbox)
        else:
            raise KeyError

    @staticmethod
    def options():
        result = "{ "
        for i in range(len(Mutation.allMutation)):
            result += f"{i} - {Mutation.allMutation[i]}, "
        result += "} = "
        return result

    @staticmethod
    def _shuffle_indexes(toolbox):
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

    @staticmethod
    def _flip_bit(toolbox):
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)
