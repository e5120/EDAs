from abc import ABCMeta, abstractmethod

import numpy as np


class EDABase(metaclass=ABCMeta):
    """
    Base class of estimation of distribution algorithms (EDAs).
    """
    def __init__(self, categories, lam, theta_init=None):
        """
        Parameters
        ----------
        categories : numpy.ndarray
            A cardinality of each dimension of the objective function.
        lam : int
            Population size.
        theta_init : numpy.ndarray, default None
            Initial probability distribution.
            The shape is (dim, max_cardinality), where max_cardinality is a maximum value in categories variable.
        """
        self.lam = lam

        self.C = categories
        self.d = len(categories)
        self.Cmax = np.max(categories)

        self.theta = np.zeros((self.d, self.Cmax))
        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1.0 / self.C[i]
        if theta_init is not None:
            self.theta = theta_init

        self.valid_params = int(np.sum(self.C - 1))
        self.valid_d = len(self.C[self.C > 1])
        # for logging
        self.best_indiv = None
        self.best_eval = np.inf     # Assume minimization problems
        self.num_evals = 0

    @abstractmethod
    def update(self, x, evals, range_restriction=False):
        """
        Build the probabilistic model.

        Parameters
        ----------
        x : array-like
            A population.
        evals : array-like
            The evaluation values in the population.
        range_restriction : bool, default False
            Whether or not to clip the probabilistic model.
        """
        pass

    def sampling(self):
        """
        Generate a individual from a probabilistic model.

        Returns
        -------
        numpy.ndarray
            A individual whose shape is (dim, Cmax).
        """
        rand = np.random.rand(self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)

        c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return c

    def convergence(self):
        """
        Measure the degree of convergence of a probabilistic model.

        Returns
        -------
        float
            Degree of the convergence of a probabilistic model.
        """
        convergence = self.theta.max(axis=1).mean()
        assert convergence <= 1.0
        return convergence

    def is_convergence(self, eps=1e-6):
        """
        Determine whether the probabilistic model has converged or not.

        Parameters
        ----------
        eps : float, default 1e-6
            Allowable error.

        Returns
        -------
        bool
            Whether the probabilistic model has converged or not.
        """
        return 1.0 - self.convergence() < eps

    def clipping(self, range_restriction):
        for i in range(self.d):
            ci = self.C[i]
            theta_min = 1.0 / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.0
            self.theta[i, :ci] = np.maximum(self.theta[i, :ci], theta_min)
            theta_sum = self.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta[i, :ci] -= (theta_sum - 1.0) * (self.theta[i, :ci] - theta_min) / tmp
            self.theta[i, :ci] /= self.theta[i, :ci].sum()

    def _preprocess(self, x, evals):
        """
        Preprocess before update method.

        Parameters
        ----------
        x : array-like
            A population.
        evals : array-like
            The evaluation values in the population.

        Returns
        -------
        numpy.ndarray
            The population sorted by their evaluation values.
        numpy.ndarray
            The evaluation values sorted by their evaluation values.
        """
        x = np.array(x)
        evals = np.array(evals)
        self.num_evals += x.shape[0]
        # sort by the evaluation value
        idx = np.argsort(evals)
        sorted_x = x[idx]
        sorted_evals = evals[idx]
        # store the best individual and the evaluation value
        if self.best_eval > sorted_evals[0]:
            self.best_eval = sorted_evals[0]
            self.best_indiv = sorted_x[0]
        return sorted_x, sorted_evals

    def __str__(self):
        return 'param size: {}\n' \
               'population size: {}\n' \
               'maximum cardinality: {}'.format(self.valid_params, self.lam, self.Cmax)
