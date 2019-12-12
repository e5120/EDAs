class CrossoverBase(object):
    def __init__(self):
        pass

    def __call__(self, parent1, parent2, fitness1, fitness2):
        return self.apply(parent1, parent2, fitness1, fitness2)

    def apply(self, parent1, parent2, fitness1, fitness2):
        raise NotImplementedError
