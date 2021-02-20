import numpy as np

from eda.objective import ObjectiveBase


class TwoMin(ObjectiveBase):
    """
    A class of two-min function.

    f(c, y) = \min(\sum_{i=1}^{D}|c_i - y_i|, \sum_{i=1}^{D}|(1 - c_i) - y_i|),
    where c is $D$-dimensional bit-strings, and y is $D$-dimensional bit-strings which is randomly generated in advance.

    Reference:
    https://arxiv.org/abs/1106.3708
    """
    def __init__(self, dim, minimize=True):
        super(TwoMin, self).__init__(dim, minimize=minimize)
        self.y = np.random.binomial(1, 0.5, dim)
        self.optimal_value = 0
        self.optimal_indiv = self.y.copy()

    def evaluate(self, c):
        c = self._check_shape(c)
        pos = np.sum(self.y != c, axis=1)
        neg = self.dim - pos
        evals = np.minimum(pos, neg)
        evals = evals if self.minimize else -evals
        info = {}
        return evals, info

    def __str__(self):
        sup_str = "    " + super(TwoMin, self).__str__().replace("\n", "\n    ")
        return 'TwoMin(\n' \
               '{}' \
               '\n)'.format(sup_str)
