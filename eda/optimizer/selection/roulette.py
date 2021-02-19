import numpy as np

from eda.optimizer.selection import SelectionBase


class Roulette(SelectionBase):
    """
    A class of roulette selection.
    """
    def __init__(self, selection_rate=0.5):
        super(Roulette, self).__init__(selection_rate)

    def apply(self, population, evals, sort=False):
        lam = population.shape[0]
        sample_lam = int(np.ceil(self.selection_rate * lam))
        prob = evals / np.sum(evals)
        selected = np.random.choice(np.arange(lam), sample_lam, p=prob)
        population = population[selected]
        evals = evals[selected]
        # if True, sort by the evaluation value
        population, evals = self.sort_by_fitness(population, evals, sort=sort)
        return population, evals

    def __str__(self):
        sup_str = "    " + super(Roulette, self).__str__().replace("\n", "\n    ")
        return 'Roulette Selection(\n' \
               '{}\n' \
               '\n)'.format(sup_str)
