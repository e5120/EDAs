import numpy as np


class SelectionBase(object):
    def __init__(self):
        pass

    def __call__(self, population, fitness, sort=False):
        return self.apply(population, fitness, sort=sort)

    def apply(self, popoulation, fitness, sort=False):
        raise NotImplementedError

    def sort_by_fitness(self, population, fitness, sort=False):
        if sort:
            idx = np.argsort(fitness)
            population = population[idx]
            fitness = fitness[idx]
        return population, fitness
