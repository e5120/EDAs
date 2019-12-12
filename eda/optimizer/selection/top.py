import numpy as np

from optimizer.selection.selection_base import SelectionBase


class Top(SelectionBase):
    def __init__(self, selection_rate=0.5):
        assert 0 < selection_rate <= 1.0
        self.selection_rate = selection_rate

    def apply(self, population, fitness, sort=False):
        lam = population.shape[0]
        sample_lam = int(np.ceil(self.selection_rate * lam))
        sorted_idx = np.argsort(fitness)[:sample_lam]
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]
        # if True, sort by fitness
        population, fitness = self.sort_by_fitness(population, fitness, sort=sort)
        return population, fitness

    def __str__(self):
        return 'Top Selection(\n' \
               '    sampling rate: {}' \
               '\n)'.format(self.selection_rate)
