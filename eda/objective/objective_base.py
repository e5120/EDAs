from abc import ABCMeta, abstractmethod

import numpy as np

from eda.utils import idx2one_hot


class ObjectiveBase(metaclass=ABCMeta):
    """
    Base class of the black-box discrete optimization problem.
    """
    def __init__(self, dim, minimize=True):
        """
        Parameters
        ----------
        dim : int
            The dimension of the problem.
        minimize : bool, default True
            Whether the problem is a minimization problem or not.
        """
        assert 0 < dim, \
            "Specify a non-negative integer."
        self.dim = dim
        self.minimize = minimize

        self.optimal_value = -np.inf if minimize else np.inf    # optimal value
        self.optimal_indiv = None   # global optimum
        self.categories = np.full(dim, 2)   # default: bit-strings

    @property
    def Cmax(self):
        return np.max(self.categories)

    def __call__(self, c, **kwargs):
        return self.evaluate(c, **kwargs)

    def __str__(self):
        return 'dim: {}\n' \
               'minimize: {}'.format(self.dim, self.minimize)

    @abstractmethod
    def evaluate(self, c):
        """
        Take an individual or a population which is group of individuals as an input, return the evaluation value of each individual.

        Parameters
        ----------
        c : array-like
            An individual or a population.

        Returns
        -------
        numpy.ndarray
            The evaluation values of the individuals.
        """
        pass

    def _check_shape(self, c):
        """
        Parameters
        ----------
        c : array-like
            An individual or a population.
            If c is an individual, assume that the shape of c is (dim, one-hot), otherwise (population_size, dim, one-hot).

        Returns
        -------
        numpy.ndarray
            A population after the input c was converted to ndarray.
            The shape is (population_size, dim).
        """
        assert isinstance(c, (list, tuple, np.ndarray)), \
            "Input c is required to be of type list, tuple, or numpy.ndarray."
        c = np.array(c)
        assert 2 <= len(c.shape) <= 3, \
            "The shape must be ({0}, {1}) or (population_size, {0}, {1}).\n\t\t" \
            "The shape of the input was {2}".format(self.dim, self.Cmax, c.shape)
        # convert an individual to a population whose size is one.
        if len(c.shape) == 2:
            c = c[np.newaxis]
        _, dim, cardinality = c.shape
        assert dim == self.dim, \
            "The dimension of an individual ({}) does not " \
            "match that of the problem ({}).\n".format(dim, self.dim)
        assert cardinality == self.Cmax, \
            "The cardinality of an individual does not match that of the problem.\n" \
            "Input({}) and problem({})".format(cardinality, self.Cmax)
        c = np.argmax(c, axis=2)
        return c

    def get_optimum(self, one_hot=False):
        """
        Return the optimum solution of the problem.

        Parameters
        ----------
        one_hot : bool, default False
            Whether or not to return the optimum with one-hot expression.

        Returns
        -------
        numpy.ndarray
           The optimum solution.
           If the optimum is not defined, return None"
        """
        if self.optimal_indiv is None:
            return None
        else:
            return idx2one_hot(self.optimal_indiv, self.Cmax) if one_hot else self.optimal_indiv
