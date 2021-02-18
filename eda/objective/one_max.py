import numpy as np

from eda.objective import ObjectiveBase


class OneMax(ObjectiveBase):
    """
    A class of one-max function.
    f(c) = \Sum_{i=1}^{D}(c_i), where c = (c_1, c_2, ..., c_D).
    """
    def __init__(self, dim, minimize=True):
        super(OneMax, self).__init__(dim, minimize=minimize)
        self.optimal_value = -dim if minimize else dim
        self.optimal_indiv = np.full(dim, 1)    # (1, 1, ..., 1)

    def evaluate(self, c):
        # The shape of c is (dim, one-hot) or (population_size, dim, one-hot)
        c = self._check_shape(c)
        # The shape of c is (population_size, dim)
        evals = np.sum(c, axis=1)
        evals = -evals if self.minimize else evals
        info = {}
        return evals, info

    def __str__(self):
        sup_str = "    " + super(OneMax, self).__str__().replace("\n", "\n    ")
        return 'OneMax(\n' \
               '{}' \
               '\n)'.format(sup_str)


if __name__=="__main__":
    from utils import idx2one_hot
    lam, dim = 2, 5
    obj = OneMax(dim)
    population = np.random.randint(0, 2, (lam, dim))
    print(population)
    population = idx2one_hot(population, 2)
    evals, _ = obj(population)
    print(evals)
