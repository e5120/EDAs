import numpy as np

from eda.optimizer.selection import SelectionBase


class Roulette(SelectionBase):
    """
    A class of roulette selection.
    """
    def __init__(self, selection_rate=0.5, criterion="eval"):
        """
        Parameters
        ----------
        criterion : str, default "eval"
            Evaluation criterion when the probability is calculated.
            Choose from ["eval", "rank"].
        """
        super(Roulette, self).__init__(selection_rate)
        assert criterion in ["eval", "rank"]
        self.criterion = criterion

    def apply(self, population, evals, sort=False):
        lam = population.shape[0]
        sample_lam = int(np.ceil(self.selection_rate * lam))
        if self.criterion == "eval":
            min_eval = np.min(evals)
            max_eval = np.max(evals)
            nor_evals = 1 - (evals - min_eval) / (max_eval - min_eval)
            prob = nor_evals / np.sum(nor_evals)
        else:
            idx = np.argsort(evals)[::-1]
            rank = np.arange(lam)
            prob = rank / np.sum(rank)
            prob = prob[np.argsort(idx)]
        selected = np.random.choice(lam, sample_lam, p=prob)
        population = population[selected]
        evals = evals[selected]
        # if True, sort by the evaluation value
        if sort:
            population, evals = self.sort_by_fitness(population, evals)
        return population, evals

    def __str__(self):
        sup_str = "    " + super(Roulette, self).__str__().replace("\n", "\n    ")
        return 'Roulette Selection(\n' \
               '{}\n' \
               '    criterion: {}' \
               '\n)'.format(sup_str, self.criterion)
