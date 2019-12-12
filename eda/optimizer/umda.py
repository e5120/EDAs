import numpy as np

from optimizer.eda_base import EDABase


class UMDA(EDABase):
    def __init__(self, categories, lr, selection, lam=64, theta_init=None):
        super(UMDA, self).__init__(categories, lam=lam, theta_init=theta_init)
        # only binary strings
        assert self.Cmax == 2
        assert 0 < lr < 1
        self.selection = selection
        self.lr = lr

    def update(self, c_one, fxc, range_restriction=False):
        self.eval_count += c_one.shape[0]
        # sort by fitness and get the index of the top of "selection_rate"%
        idx = np.argsort(fxc)
        # store best individual and evaluation value
        if self.best_eval > fxc[idx[0]]:
            self.best_eval = fxc[idx[0]]
            self.best_indiv = c_one[idx[0]]
        # apply selection
        if self.selection is not None:
            c_one, fxc = self.selection.apply(c_one, fxc)
        # update probability vector
        self.theta[:, -1] += self.lr * (np.mean(c_one[:, :, -1], axis=0) - self.theta[:, -1])
        self.theta[:, 0] = 1 - self.theta[:, -1]
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
        sup_str = "    " + super(UMDA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        return 'UMDA(\n' \
               '{}\n' \
               '{}\n' \
               '    lr: {}\n' \
               ')'.format(sup_str, sel_str, self.lr)
