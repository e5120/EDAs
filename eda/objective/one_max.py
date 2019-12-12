import numpy as np

from objective.objective_base import ObjectiveBase


class OneMax(ObjectiveBase):
    def __init__(self, D, target, K=2, noise=False):
        super(OneMax, self).__init__(D, K, target, noise=noise)

    def forward(self, c):
        c = np.argmax(c, axis=2)
        loss = -np.sum(c, axis=1)
        general_loss = loss
        return loss, general_loss, {}

    def __str__(self):
        sup_str = "    " + super(OneMax, self).__str__().replace("\n", "\n    ")
        return 'OneMax(\n' \
               '{}' \
               '\n)'.format(sup_str)
