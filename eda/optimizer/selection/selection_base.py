import numpy as np


class SelectionBase(object):
    def __init__(self):
        pass

    def __call__(self, population, fitness):
        return self.apply(population, fitness)

    def apply(self, popoulation, fitness):
        raise NotImplementedError

    def sort_by_fitness(self, population, fitness, sort=False):
        if sort:
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
        return population, fitness
