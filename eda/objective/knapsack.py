import numpy as np

from objective.objective_base import ObjectiveBase


class Knapsack(ObjectiveBase):
    def __init__(self, K, D, C, items, target, noise=False):
        super(Knapsack, self).__init__(D, K, target, noise=noise)
        self.C = C
        self.items = items
        self.values = np.array([item.value for item in items])
        self.weights = np.array([item.weight for item in items])
        if K == -1:
            self.categories = np.array(list(map(lambda x: int(x),
                                                C / np.array([item.weight for item in items])))) + 1

    def forward(self, c):
        c = np.argmax(c, axis=2)
        # 制約(指定容積未満かどうか)を満たすかチェック
        loss = np.where(np.dot(c, self.weights) <= self.C,
                        -np.dot(c, self.values),
                        2**32)
        general_loss = loss
        return loss, general_loss, {}

    def __str__(self):
        sup_str = "    " + super(Knapsack, self).__str__().replace("\n", "\n    ")
        return 'Knapsack(\n' \
               '{}\n' \
               ')\n'.format(sup_str)
