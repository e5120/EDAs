class ReplacementBase(object):
    def __init__(self):
        pass

    def __call__(self, population, p_fitness, candidates, c_fitness):
        return self.replace(population, p_fitness, candidates, c_fitness)

    def replace(self, popoulation, p_fitness, candidates, c_fitness):
        raise NotImplementedError
