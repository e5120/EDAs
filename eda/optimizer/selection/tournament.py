import numpy as np

from eda.optimizer.selection import SelectionBase


class Tournament(SelectionBase):
    """
    A class of tournament selection.
    """
    def __init__(self, selection_rate=0.5, k=2, replace=False):
        """
        Parameters
        ----------
        k : int, default 2
            Tournament size.
        replace : bool, default False
            Sampling with replacement or not.
        """
        super(Tournament, self).__init__(selection_rate)
        if not replace:
            assert 0.0 < selection_rate <= 1 / k
        self.k = k
        self.replace = replace

    def apply(self, population, evals, sort=False):
        assert self.k <= population.shape[0]
        lam = population.shape[0]
        sample_lam = int(self.selection_rate * lam)
        # get population for Tournament
        if self.replace:
            sample_idx = np.array(
                [np.random.choice(lam, self.k, replace=False) for _ in range(sample_lam)]
            )
        else:
            sample_idx = np.random.choice(lam, (sample_lam, self.k), replace=False)
        # get winner index of each tournament
        winner_idx = np.argmin(evals[sample_idx], axis=1)
        winner_idx = [sample_idx[i, winner_idx[i]] for i in range(len(winner_idx))]
        # select population for next generation
        population = population[winner_idx]
        evals = evals[winner_idx]
        # if True, sort by the evaluations
        if sort:
            population, evals = self.sort_by_fitness(population, evals)
        return population, evals

    def __str__(self):
        sup_str = "    " + super(Tournament, self).__str__().replace("\n", "\n    ")
        return 'Tournament Selection(\n' \
               '{}\n' \
               '    tournament size: {}\n' \
               '    with replacement: {}' \
               '\n)'.format(sup_str, self.k, self.replace)
