import numpy as np

from eda.optimizer.eda_base import EDABase


class UMDA(EDABase):
    """
    A class of univariate marginal distribution algorithm (UMDA)
    """
    def __init__(self, categories, lr, selection, lam=64, theta_init=None):
        super(UMDA, self).__init__(categories, lam=lam, theta_init=theta_init)
        """
        Parameters
        ----------
        lr : float
            Learning rate.
        selection : eda.optimizer.selection.selection_base.SelectionBase
            Selection method.
        """
        # only binary strings
        assert self.Cmax == 2
        assert 0 < lr < 1
        self.selection = selection
        self.lr = lr

    def update(self, x, evals, range_restriction=False):
        x, evals = self._preprocess(x, evals)
        # apply selection
        if self.selection is not None:
            x, evals = self.selection(x, evals)
        # update probability vector
        self.theta[:, -1] += self.lr * (np.mean(x[:, :, -1], axis=0) - self.theta[:, -1])
        self.theta[:, 0] = 1 - self.theta[:, -1]
        # if range_restriction is True, clipping
        self.clipping(range_restriction)

    def __str__(self):
        sup_str = "    " + super(UMDA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        return 'UMDA(\n' \
               '{}\n' \
               '    lr: {}\n' \
               '{}\n' \
               ')'.format(sup_str, self.lr, sel_str)
