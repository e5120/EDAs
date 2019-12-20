import numpy as np

from optimizer.eda_base import EDABase


class CGA(EDABase):
    def __init__(self, categories, lam=32, theta_init=None):
        super(CGA, self).__init__(categories, lam=2, theta_init=theta_init)
        self.all_lam = lam

    def update(self, c_one, fxc, range_restriction=False):
        self.eval_count += c_one.shape[0]
        # sort by fitness
        idx = np.argsort(fxc)
        # store best individual and evaluation value
        if self.best_eval > fxc[idx[0]]:
            self.best_eval = fxc[idx[0]]
            self.best_indiv = c_one[idx[0]]
        # get winner and loser, and transform one-hot vector to index
        win = np.argmax(c_one[idx[0]], axis=1)
        lose = np.argmax(c_one[idx[-1]], axis=1)
        # get index for updating parameter that is difference between winner and loser
        diff_idx = win != lose
        # update parameter
        self.theta[diff_idx, win[diff_idx]] += 1 / self.all_lam
        self.theta[diff_idx, lose[diff_idx]] -= 1 / self.all_lam
        # if range_restriction is True, clipping
        for i in range(self.d):
            ci = self.C[i]
            theta_min = 1.0 / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.0
            self.theta[i, :ci] = np.maximum(self.theta[i, :ci], theta_min)
            theta_sum = self.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta[i, :ci] -= (theta_sum - 1.0) * (self.theta[i, :ci] - theta_min) / tmp
            self.theta[i, :ci] /= self.theta[i, :ci].sum()

    def __str__(self):
        sup_str = "    " + super(CGA, self).__str__().replace("\n", "\n    ")
        return 'CGA(\n' \
               '{}' \
               '\n)'.format(sup_str)
