import numpy as np

from optimizer.mutation.mutation_base import MutationBase


class Mutation(MutationBase):
    def __init__(self, prob):
        self.prob = prob

    def apply(self, population):
        assert population.shape[2] == 2
        mut_idx = np.random.rand(population.shape[0], population.shape[1]) <= self.prob
        population[mut_idx] = np.logical_not(population[mut_idx]).astype(np.int32)
        return population

    def __str__(self):
        return 'Mutation(\n' \
                '    mutation prob: {}\n' \
                ')\n'.format(self.prob)
