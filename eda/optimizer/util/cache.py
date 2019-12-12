import numpy as np


class Cache(object):
    def __init__(self, dim):
        self.dim = dim
        self.init()

    def add(self, i, j, delta_cc, subset):
        self.cc_list[i, j] = delta_cc
        self.subsets[i, j] = subset

    def remove(self, idx):
        self.cc_list = np.delete(self.cc_list, idx, 0)
        self.cc_list = np.delete(self.cc_list, idx, 1)
        self.subsets = np.delete(self.subsets, idx, 0)
        self.subsets = np.delete(self.subsets, idx, 1)

    def argmax_cc(self):
        return np.unravel_index(np.argmax(self.cc_list), self.cc_list.shape)

    def init(self):
        self.cc_list = np.zeros((self.dim, self.dim))
        self.subsets = np.zeros((self.dim, self.dim), dtype=object)
