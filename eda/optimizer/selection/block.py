import numpy as np

from optimizer.selection.selection_base import SelectionBase


class Block(SelectionBase):
    def __init__(self, sampling_rate=0.5):
        self.sampling_rate = sampling_rate

    def apply(self, population, fitness, sort=False):
        lam = population.shape[0]
        idx = np.argsort(fitness)[:int(self.sampling_rate*lam)]
        population = population[idx]
        fitness = fitness[idx]
        # duplicate top of population in term of fitness
        dup_num = int(np.ceil(1 / self.sampling_rate))
        population = np.tile(population, (dup_num, 1, 1))
        fitness = np.tile(fitness, dup_num)
        if population.shape[0] > lam:
            population = population[:lam]
            fitness = fitness[:lam]
        # if True, sort by fitness
        population, fitness = self.sort_by_fitness(population, fitness, sort=sort)
        return population, fitness

    def __str__(self):
        return 'Block Selection(\n' \
                '    sampling rate: {}' \
                '\n)'.format(self.sampling_rate)
