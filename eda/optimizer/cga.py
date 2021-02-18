import numpy as np

from eda.optimizer.eda_base import EDABase


class CGA(EDABase):
    """
    A class of compact genetic algorithm (cGA)
    """
    def __init__(self, categories, lam=32, theta_init=None):
        super(CGA, self).__init__(categories, lam=2, theta_init=theta_init)
        self.diff = 1.0 / lam

    def update(self, x, evals, range_restriction=False):
        assert x.shape[0] == 2
        x, evals = self._preprocess(x, evals)
        # sort by the evaluation value
        idx = np.argsort(evals)
        # get winner and loser, and transform one-hot vector to index
        win = np.argmax(x[idx[0]], axis=1)
        lose = np.argmax(x[idx[-1]], axis=1)
        # get indexes to update parameters, where the indexes are
        # the dimensions that have different values between winner and loser
        diff_idx = win != lose
        # update parameter
        self.theta[diff_idx, win[diff_idx]] += self.diff
        self.theta[diff_idx, lose[diff_idx]] -= self.diff
        # if range_restriction is True, clipping the probabilistic model
        self.clipping(range_restriction)

    def __str__(self):
        sup_str = "    " + super(CGA, self).__str__().replace("\n", "\n    ")
        return 'cGA(\n' \
               '{}' \
               '\n)'.format(sup_str)
