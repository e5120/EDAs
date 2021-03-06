import numpy as np

from optimizer.selection.selection_base import SelectionBase


class Tournament(SelectionBase):
    def __init__(self, k=2, sampling_rate=1.0, replace=True):
        assert 0.0 < sampling_rate <= 1.0
        if not replace:
            assert 0.0 < sampling_rate <= 1 / k
        self.k = k
        self.sampling_rate = sampling_rate
        self.replace = replace

    def apply(self, population, fitness, sort=False):
        assert self.k <= population.shape[0]
        lam = population.shape[0]
        sample_lam = int(self.sampling_rate * lam)
        # get population for Tournament
        if self.replace:
            sample_idx = np.array(
                [np.random.choice(lam, self.k, replace=False) for _ in range(sample_lam)]
            )
        else:
            sample_idx = np.random.choice(lam, (sample_lam, self.k), replace=False)
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
               '    tournament size: {}\n' \
               '    sampling rate: {}\n' \
               '    with replacement: {}' \
               '\n)'.format(self.k, self.sampling_rate, self.replace)
