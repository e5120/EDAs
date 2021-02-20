import numpy as np

from eda.optimizer.selection import SelectionBase


class Block(SelectionBase):
    """
    A class of block selection.
    """
    def __init__(self, selection_rate=0.5):
        super(Block, self).__init__(selection_rate)

    def apply(self, population, evals, sort=False):
        lam = population.shape[0]
        idx = np.argsort(evals)[:int(self.selection_rate*lam)]
        population = population[idx]
        evals = evals[idx]
        # duplicate top individuals in a population based on the evaluation value
        dup_num = int(np.ceil(1 / self.selection_rate))
        pop_shape = np.ones(len(population.shape), dtype=np.int)
        pop_shape[0] *= dup_num
        population = np.tile(population, pop_shape)
        evals = np.tile(evals, dup_num)
        if population.shape[0] > lam:
            population = population[:lam]
            evals = evals[:lam]
        # if True, sort by the evaluation value
        if sort:
            population, evals = self.sort_by_fitness(population, evals)
        return population, evals

    def __str__(self):
        sup_str = "    " + super(Block, self).__str__().replace("\n", "\n    ")
        return 'Block Selection(\n' \
               '{}' \
               '\n)'.format(sup_str)
