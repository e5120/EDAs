from abc import ABCMeta, abstractmethod

import numpy as np


class MetricBase(metaclass=ABCMeta):
    """
    Base class of evaluation metrics which measure dependencies among variables.
    """
    def __init__(self, data, base):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Dataset
        base : numpy.ndarray
            Cardinality of each dimension
        """
        assert len(data.shape) == 2
        assert np.all(base > 0)
        assert len(base.shape) == 1 and base.shape[0] == data.shape[1]
        self.data = data
        self.data_size = data.shape[0]
        self.dim = data.shape[1]
        self.base = base

    def score(self, network):
        """
        Calculate a score of a network.

        Parameters
        ----------
        network : numpy.ndarray
            A Bayesian network which is represented by adjacency matrix.

        Returns
        -------
        float
            Evaluation value after logarithmic transformation.
        """
        assert len(network.shape) == 2 and network.shape[0] == network.shape[1] and network.shape[0] == self.dim
        score = 0
        for node in range(self.dim):
            parents = network[node] != 0
            score += self.local_score(node, parents)
        score += self.structure_prior(network)
        return score

    @abstractmethod
    def local_score(self, node, parents):
        """
        Calculate a score of each node.

        Parameters
        ----------
        node : int
            Node number.
        parents : numpy.ndarray
            A set of parent nodes of the node.
        """
        pass

    def structure_prior(self, network):
        """
        Return a prior distribution after logarithmic transformation.
        If you have prior information, override this method.

        Parameters
        ----------
        network : numpy.ndarray
            A Bayesian network which is represented by adjacency matrix.

        Returns
        -------
        float
            Prior distribution after logarithmic transformation.
        """
        return 0
