import numpy as np

from eda.objective import ObjectiveBase


class DeceptiveTrap(ObjectiveBase):
    """
    A class of Deceptive-k Trap function.
    A user parameter k determines the number of dependencies among variables.
    A user parameter d determines the deceptiveness of the problem.
    When k=3 and d=0.1, the evaluation value is calculated as
    f(c) = \Sum_{i=0}^{D/3-1}g(c_{3i+1},c_{3i+2},c_{3i+3}),
    g(c_1,c_2,c_3) = 1-d,  i.e., 0.9 if \Sum_{j}c_j = 0,
                     1-2d, i.e., 0.8 if \Sum_{j}c_j = 1,
                     0               if \Sum_{j}c_j = 2,
                     1               if \Sum_{j}c_j = 3,
    where c = (c_1, c_2, ..., c_D).

    Reference:
    https://dl.acm.org/doi/pdf/10.5555/2933923.2933973
    """
    def __init__(self, dim, k=3, d=0.1, minimize=True):
        """
        Parameters
        ----------
        k : int, default 3
            The number of dependencies among variables.
        d : float, default 0.1
            A user parameter which determines the deceptiveness of the problem.
        """
        super(DeceptiveTrap, self).__init__(dim, minimize=minimize)
        assert dim % k == 0
        assert (k - 1) * d < 1
        self.k = k
        self.d = d

        self.optimal_value = -dim / k if minimize else dim / k
        self.optimal_indiv = np.ones(dim, dtype=np.int)

    def evaluate(self, c):
        c = self._check_shape(c)
        c = c.reshape(c.shape[0], -1, self.k)
        c = np.sum(c, axis=2)
        evals = np.zeros((c.shape[0], c.shape[1]))
        for i in range(self.k - 1):
            evals = np.where(c == i,
                             1.0 - (i + 1) * self.d,
                             evals)
        evals = np.where(c == self.k, 1.0, evals)
        evals = np.sum(evals, axis=1)
        evals = -evals if self.minimize else evals
        info = {}
        return evals, info

    def __str__(self):
        sup_str = "    " + super(DeceptiveTrap, self).__str__().replace("\n", "\n    ")
        return 'Deceptive-k Trap(\n' \
               '{}\n' \
               '    k: {}\n' \
               '    d: {}\n' \
               ')\n'.format(sup_str, self.k, self.d)
