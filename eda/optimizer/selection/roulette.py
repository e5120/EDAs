import numpy as np

from optimizer.selection.selection_base import SelectionBase


class Roulette(SelectionBase):
    def __init__(self, selection_rate=0.5):
        assert 0 < selection_rate <= 1.0
        self.selection_rate = selection_rate

    def apply(self, population, fitness, sort=False):
        lam = population.shape[0]
        sample_lam = int(np.ceil(self.selection_rate * lam))
        prob = fitness / np.sum(fitness)
        selected = np.random.choice(np.arange(lam), sample_lam, p=prob)
        population = population[selected]
        fitness = fitness[selected]
        # if True, sort by fitness
        population, fitness = self.sort_by_fitness(population, fitness, sort=sort)
        return population, fitness

    def __str__(self):
        return 'Roulette Selection(\n' \
               '    sampling rate: {}' \
               '\n)'.format(self.selection_rate)
