import numpy as np
from scipy.special import gammaln

from eda.optimizer.metric import MetricBase


class K2(MetricBase):
    """
    A class of K2 metric.
    """
    def __init__(self, data, base):
        super(K2, self).__init__(data, base)

    def local_score(self, node, parents):
        score = 0
        # root node
        if not np.any(parents):
            _, n_counts = np.unique(self.data[:, node], return_counts=True, axis=0)
            score += gammaln(self.base[node]) - np.sum(gammaln(self.data_size + self.base[node])) + np.sum(gammaln(n_counts + 1))
        # others
        else:
            p, p_counts = np.unique(self.data[:, parents], return_counts=True, axis=0)
            _, n_p_counts = np.unique(np.c_[self.data[:, node], self.data[:, parents]], return_counts=True, axis=0)
            score += gammaln(self.base[node]) * len(p_counts) - np.sum(gammaln(p_counts + self.base[node])) + np.sum(gammaln(n_p_counts + 1))
        return score
