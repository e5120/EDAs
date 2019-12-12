import numpy as np

from objective.objective_base import ObjectiveBase


class DeceptiveOrder3(ObjectiveBase):
    def __init__(self, D, target, d=0.1, K=2, noise=False):
        super(DeceptiveOrder3, self).__init__(D, K, target, noise=noise)
        self.d = d

    def forward(self, c):
        c = np.argmax(c, axis=2)
        c = c.reshape(c.shape[0], -1, 3)
        c = np.sum(c, axis=2)
        loss = np.where(c == 0, 1 - self.d, 0)
        loss = np.where(c == 1, 1 - 2 * self.d, loss)
        loss = np.where(c == 3, 1, loss)
        loss = -np.sum(loss, axis=1)
        general_loss = loss
        return loss, general_loss, {}

    def __str__(self):
        sup_str = "    " + super(DeceptiveOrder3, self).__str__().replace("\n", "\n    ")
        return 'Deceptive Order 3(\n' \
               '{}\n' \
               '    subtraction value: {}\n' \
               ')\n'.format(sup_str, self.d)
