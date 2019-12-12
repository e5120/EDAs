class CrossoverBase(object):
    def __init__(self):
        pass

    def apply(self, parent1, parent2, fitness1, fitness2):
        raise NotImplementedError
