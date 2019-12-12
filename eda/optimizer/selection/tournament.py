import numpy as np

from optimizer.selection.selection_base import SelectionBase


class Tournament(SelectionBase):
    def __init__(self, sampling_rate=0.5):
        assert 0 < sampling_rate <= 1.0
        self.sampling_rate = sampling_rate

    def apply(self, population, fitness, sort=False):
        lam = population.shape[0]
        sample_lam = int(np.ceil(self.sampling_rate * lam))
        target_idx = np.arange(lam)
        # get population for Tournament
        sample_idx = np.array(
            [np.random.choice(target_idx, sample_lam, replace=False) for _ in range(lam)]
        )
        # get winner index of each tournament
        winner_idx = np.argmin(fitness[sample_idx], axis=1)
        winner_idx = [sample_idx[i, winner_idx[i]] for i in range(len(winner_idx))]
        # select population for next generation
        population = population[winner_idx]
        fitness = fitness[winner_idx]
        # if True, sort by fitness
        population, fitness = self.sort_by_fitness(population, fitness, sort=sort)
        return population, fitness

    def __str__(self):
        return 'Tournament Selection(\n' \
               '    sampling rate: {}' \
               '\n)'.format(self.sampling_rate)
