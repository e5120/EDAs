import numpy as np

from optimizer.replacement.replacement_base import ReplacementBase


class RestrictedReplacement(ReplacementBase):
    def __init__(self, window_size, dim):
        ws = int(dim / 20)
        self.window_size = ws if ws > 1 and window_size > ws else window_size

    def replace(self, population, p_fitness, candidates, c_fitness):
        assert population.shape[1] == candidates.shape[1]
        lam = population.shape[0]
        lam_c = candidates.shape[0]
        sampled_idx = np.random.randint(0, lam, (self.window_size, lam_c))
        distances = np.sum(population[sampled_idx] != candidates, axis=2)
        target_idx = sampled_idx[np.argmin(distances, axis=0), np.arange(lam)]
        for c_idx, p_idx in enumerate(target_idx):
            if p_fitness[p_idx] > c_fitness[c_idx]:
                p_fitness[p_idx] = c_fitness[c_idx]
                population[p_idx] = candidates[c_idx]
        return population, p_fitness

    def __str__(self):
        return 'Restricted Replacement(\n' \
               '    window size: {}' \
               '\n)'.format(self.window_size)
