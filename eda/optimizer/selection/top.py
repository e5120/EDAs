import numpy as np

from eda.optimizer.selection import SelectionBase


class Top(SelectionBase):
    """
    A class of top selection.
    """
    def __init__(self, selection_rate=0.5):
        super(Top, self).__init__(selection_rate)

    def apply(self, population, evals, sort=False):
        lam = population.shape[0]
        sample_lam = int(np.ceil(self.selection_rate * lam))
        sorted_idx = np.argsort(evals)[:sample_lam]
        population = population[sorted_idx]
        evals = evals[sorted_idx]
        # if True, sort by the evaluation value
        if sort:
            population, evals = self.sort_by_fitness(population, evals)
        return population, evals

    def __str__(self):
        sup_str = "    " + super(Top, self).__str__().replace("\n", "\n    ")
        return 'Top Selection(\n' \
               '{}' \
               '\n)'.format(sup_str)
