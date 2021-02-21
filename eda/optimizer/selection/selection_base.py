from abc import ABCMeta, abstractmethod

import numpy as np


class SelectionBase(metaclass=ABCMeta):
    """
    Base class of selection methods.
    """
    def __init__(self, selection_rate):
        """
        Parameters
        ----------
        selection_rate : float
            Selection rate, i.e., how many individuals are chosen when the selection method is applied to a population.
        """
        assert 0 < selection_rate <= 1.0
        self.selection_rate = selection_rate

    def __call__(self, population, evals, sort=False):
        return self.apply(population, evals, sort=sort)

    @abstractmethod
    def apply(self, population, evals, sort=False):
        """
        Apply selection to a population.

        Parameters
        ----------
        population : numpy.ndarray
            Population.
        evals : numpy.ndarray
            Evaluation values corresponding to individuals in a population.
        sort : bool, default False
            Whether or not to sort a population by the evaluation value.

        Returns
        -------
        numpy.ndarray
            A population which includes chosen individuals of the input popoulation.
        numpy.ndarray
            The evaluation values corresponding to the above population.
        """
        pass

    def sort_by_fitness(self, population, evals):
        """
        Sort by the evaluation value.
        """
        idx = np.argsort(evals)
        population = population[idx]
        evals = evals[idx]
        return population, evals

    def __str__(self):
        return 'selection rate: {}'.format(self.selection_rate)
