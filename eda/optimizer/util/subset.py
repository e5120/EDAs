import copy

import numpy as np
from scipy import stats


class SubSet(object):
    def __init__(self, idx, indiv, c_max):
        self.idx_set = [idx]
        self.lam = indiv.shape[0]
        self.indiv = [indiv]
        self.c_max = c_max
        self.theta = np.array([np.sum(indiv == i) for i in range(self.c_max)]) / self.lam
        self.set_cc()

    def merge(self, other):
        assert self.lam == other.lam
        merged = copy.deepcopy(self)
        merged.idx_set += other.idx_set
        merged.indiv += other.indiv
        merged.theta = np.zeros([self.c_max for _ in range(len(merged))])
        freqes = np.stack(merged.indiv, axis=1)
        pattern, count = np.unique(freqes, axis=0, return_counts=True)
        for p, c in zip(pattern, count):
            merged.theta[tuple(p)] = c
        merged.theta /= merged.lam
        merged.set_cc()
        return merged

    def set_cc(self):
        self.entropy = stats.entropy(self.theta.flatten(), base=self.c_max)
        self.mc = np.log(self.lam) / np.log(self.c_max) * np.power(2, len(self))
        self.cpc = self.lam * self.entropy
        self.cc = self.mc + self.cpc

    def __len__(self):
        return len(self.idx_set)

    def __str__(self):
        return "index : {}, theta : {}, entropy : {}".format(
                self.idx_set, self.theta, self.entropy)

    def __format__(self, spec):
        if spec == "idx":
            return "index_set : {}".format(self.idx_set)
        elif spec == "theta":
            return "theta : {}".format(self.theta)
        elif spec == "entropy":
            return "entropy : {}".format(self.entropy)
