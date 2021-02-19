import numpy as np

from eda.optimizer.eda_base import EDABase


class PBIL(EDABase):
    """
    A class of population-based incremental learning (PBIL)
    """
    def __init__(self, categories, lr, lam=64, negative_lr=None,
                 mut_prob=0.02, mut_shift=0.05, theta_init=None):
        super(PBIL, self).__init__(categories, lam=lam, theta_init=theta_init)
        """
        Parameters
        ----------
        lr : float
            Learning rate.
        negative_lr : float, default None
            Learning rate for negative example.
            if None, disable the use of negative examples.
        mut_prob : float, default 0.02
            Mutation probability.
        mut_shift : float, default 0.05
            Amount of shift in mutation.
        """
        # only binary strings
        assert self.Cmax == 2
        assert 0.0 < lr < 1.0
        assert negative_lr is None or 0.0 < negative_lr < 1.0
        assert 0.0 <= mut_prob <= 1.0
        self.lr = lr
        self.negative_lr = negative_lr
        self.mut_prob = mut_prob
        self.mut_shift = mut_shift

    def update(self, x, evals, range_restriction=False):
        x, evals = self._preprocess(x, evals)
        # update probability vector
        best_indiv = x[0]
        self.theta[:, -1] = (1.0 - self.lr) * self.theta[:, -1] + self.lr * best_indiv[:, -1]
        # update probability vector by negative example
        if self.negative_lr is not None:
            worst_indiv = x[-1]
            diff_dix = best_indiv[:, -1] != worst_indiv[:, -1]
            self.theta[diff_dix, -1] = (1.0 - self.negative_lr) * self.theta[diff_dix, -1] + self.negative_lr * best_indiv[diff_dix, -1]
        # mutation probability vector
        mut_idx = np.random.rand(self.d) < self.mut_prob
        mut_num = np.sum(mut_idx)
        self.theta[mut_idx, -1] = (1.0 - self.mut_shift) * self.theta[mut_idx, -1] \
                                  + np.random.randint(0, 2, mut_num) * self.mut_shift
        self.theta[:, 0] = 1 - self.theta[:, -1]
        # if range_restriction is True, clipping
        self.clipping(range_restriction)

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
