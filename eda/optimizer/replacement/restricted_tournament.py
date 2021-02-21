import numpy as np

from eda.optimizer.replacement import ReplacementBase


class RestrictedTournament(ReplacementBase):
    """
    A class of restricted tournament replacement (RTR).
    """
    def __init__(self, dim, replace_rate=0.5, window_size=2, fix_size=True):
        super(RestrictedTournament, self).__init__(replace_rate, fix_size=fix_size)
        """
        Parameters
        ----------
        dim : int
            The dimension of the problem.
        window_size : int, default 2
            A user parameter which determines trade-off between the goodness and the diversity in the population.
        """
        assert 0 < dim
        ws = int(dim / 20)
        self.window_size = ws if ws > 1 and window_size > ws else window_size

    def apply(self, parent, p_evals, candidate, c_evals):
        p_lam = parent.shape[0]
        c_lam = candidate.shape[0]
        replaced_lam = int(p_lam * self.replace_rate)
        assert replaced_lam == c_lam, \
            "The number of individuals for the replacement({}) must match the population size of candidate({})".format(replaced_lam, c_lam)
        sampled_idx = np.random.randint(0, p_lam, (self.window_size, c_lam))
        # In the case of (population_size, dim)
        if len(parent.shape) == 2:
            distances = np.sum(parent[sampled_idx] != candidate, axis=2)
        # In the case of (population_size, dim, one_hot)
        elif len(parent.shape) == 3:
            distances = np.sum(np.argmax(parent[sampled_idx], axis=3) != np.argmax(candidate, axis=2), axis=2)
        else:
            print("error. shape of ndarray is wrong")
            exit()
        target_idx = sampled_idx[np.argmin(distances, axis=0), np.arange(c_lam)]
        for c_idx, p_idx in enumerate(target_idx):
            if p_evals[p_idx] > c_evals[c_idx]:
                p_evals[p_idx] = c_evals[c_idx]
                parent[p_idx] = candidate[c_idx]
        return parent, p_evals

    def __str__(self):
        sup_str = "    " + super(RestrictedTournament, self).__str__().replace("\n", "\n    ")
        return 'Restricted Tournament Replacement(\n' \
               '{}\n' \
               '    window size: {}' \
               '\n)'.format(sup_str, self.window_size)
