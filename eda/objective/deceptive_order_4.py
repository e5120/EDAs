import numpy as np

from objective.objective_base import ObjectiveBase


class DeceptiveOrder4(ObjectiveBase):
    def __init__(self, D, target, K=2, noise=False):
        super(DeceptiveOrder4, self).__init__(D, K, target, noise=noise)
        self.eval_table = {
            "[1 1 1 1]": 30,
            "[0 0 0 0]": 28,
            "[0 0 0 1]": 26,
            "[0 0 1 0]": 24,
            "[0 1 0 0]": 22,
            "[1 0 0 0]": 20,
            "[0 0 1 1]": 18,
            "[0 1 0 1]": 16,
            "[0 1 1 0]": 14,
            "[1 0 0 1]": 12,
            "[1 0 1 0]": 10,
            "[1 1 0 0]": 8,
            "[1 1 1 0]": 6,
            "[1 1 0 1]": 4,
            "[1 0 1 1]": 2,
            "[0 1 1 1]": 0,
        }

    def forward(self, c):
        c = np.argmax(c, axis=2)
        c = c.reshape(c.shape[0], -1, 4)
        loss = np.array([list(map(lambda x: self.eval_table[str(x)], _c)) for _c in c])
        loss = -np.sum(loss, axis=1)
        general_loss = loss
        bit, count = np.unique(c[np.argmin(loss)], return_counts=True, axis=0)
        info = {"type": bit, "freq": count}
        return loss, general_loss, info

    def __str__(self):
        sup_str = "    " + super(DeceptiveOrder4, self).__str__().replace("\n", "\n    ")
        return 'Deceptive Order 4(\n' \
               '{}' \
               '\n)'.format(sup_str)
