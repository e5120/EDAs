import numpy as np

from optimizer.replacement.replacement_base import ReplacementBase


class TruncatedReplacement(ReplacementBase):
    def __init__(self):
        pass

    def replace(self, population, p_fitness, candidates, c_fitness):
        assert population.shape[1] == candidates.shape[1]
        lam_c = candidates.shape[0]
        sort_idx = np.argsort(p_fitness)
        population = population[sort_idx]
        p_fitness = p_fitness[sort_idx]
        population[-lam_c:] = candidates
        p_fitness[-lam_c:] = c_fitness
        return population, p_fitness

    def __str__(self):
        return 'Truncated Replacement(\n)'
