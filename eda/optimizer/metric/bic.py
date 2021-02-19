import numpy as np

from eda.optimizer.metric import MetricBase
from eda.utils import conditional_entropy


class BIC(MetricBase):
    """
    A class of bayesian information criterion (BIC).
    """
    def __init__(self, data, base):
        super(BIC, self).__init__(data, base)

    def local_score(self, node, parents):
        conditional_entropy_term = conditional_entropy(self.data[:, node], self.data[:, parents]) * self.data_size
        regularization_term = (self.base[node] - 1) * np.prod(self.base[parents]) * np.log(self.data_size) / 2
        return -conditional_entropy_term - regularization_term
