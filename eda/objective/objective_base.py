import numpy as np


class ObjectiveBase(object):
    def __init__(self, D, K, target, noise=False):
        self.D = D
        self.K = K
        self.target = target
        self.noise = noise
        self.categories = K * np.ones(D, dtype=np.int)

    def __call__(self, c):
        return self.forward(c)

    def forward(self, c):
        return NotImplementedError

    def __str__(self):
        return 'dim: {}\n' \
               'noise: {}'.format(self.D, self.noise)
