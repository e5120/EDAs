import numpy as np

from optimizer.eda_base import EDABase


class PBIL(EDABase):
    def __init__(self, categories, lr, lam=64, negative_lr=None,
                 mut_prob=0.02, mut_shift=0.05, theta_init=None):
        super(PBIL, self).__init__(categories, lam=lam, theta_init=theta_init)
        # only binary strings
        assert self.Cmax == 2
        assert 0.0 < lr < 1.0
        assert negative_lr is None or 0.0 < negative_lr < 1.0
        assert 0.0 <= mut_prob <= 1.0
        
        self.lr = lr
        self.negative_lr = negative_lr
        self.mut_prob = mut_prob
        self.mut_shift = mut_shift

    def update(self, c_one, fxc, range_restriction=False):
        self.eval_count += c_one.shape[0]
        # sort by fitness
        idx = np.argsort(fxc)
        best_idx = idx[0]
        # store best individual and evaluation value
        if self.best_eval > fxc[best_idx]:
            self.best_eval = fxc[best_idx]
            self.best_indiv = c_one[best_idx]
        # update probability vector
        best_indiv = c_one[best_idx]
        self.theta[:, -1] = (1.0 - self.lr) * self.theta[:, -1] + self.lr * best_indiv[:, -1]
        # update probability vector by negative sample
        if self.negative_lr is not None:
            worst_idx = idx[-1]
            worst_indiv = c_one[worst_idx]
            diff_dix = best_indiv[:, -1] != worst_indiv[:, -1]
            self.theta[diff_dix, -1] = (1.0 - self.negative_lr) * self.theta[diff_dix, -1] + self.negative_lr * best_indiv[diff_dix, -1]
        # mutation probability vector
        dim = self.theta.shape[0]
        mut_idx = np.random.rand(dim) < self.mut_prob
        mut_num = np.sum(mut_idx)
        self.theta[mut_idx, -1] = (1.0 - self.mut_shift) * self.theta[mut_idx, -1] \
                                  + np.random.randint(0, 2, mut_num) * self.mut_shift
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
        sup_str = "    " + super(PBIL, self).__str__().replace("\n", "\n    ")
        return 'PBIL(\n' \
               '{}\n' \
               '    lr: {}\n' \
               '    negative lr: {}\n' \
               '    mutation prob: {}\n' \
               '    mutation shift: {}\n' \
               ')'.format(sup_str, self.lr, self.negative_lr,
                          self.mut_prob, self.mut_shift)
