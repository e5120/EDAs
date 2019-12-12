import copy

import numpy as np
from scipy import stats


class SubSet(object):
    def __init__(self, idx, indiv, c_max):
        self.idx_set = [idx]
        self.n = indiv.shape[0]
        self.indiv = [indiv]
        self.theta = np.array([np.sum(indiv == i) for i in range(c_max)]) / self.n
        self.entropy = stats.entropy(self.theta.flatten(), base=c_max)
        self.mc = np.log(self.n) / np.log(c_max) * np.power(2, len(self))
        self.cpc = self.n * self.entropy
        self.cc = self.mc + self.cpc

    def merge(self, other, c_max):
        assert self.n == other.n
        merge_tmp = copy.deepcopy(self)
        merge_tmp.idx_set += other.idx_set
        merge_tmp.indiv += other.indiv
        merge_tmp.theta = np.zeros([c_max for _ in range(len(merge_tmp))])
        freqes = np.stack(merge_tmp.indiv, axis=1)
        pattern, count = np.unique(freqes, axis=0, return_counts=True)
        for p, c in zip(pattern, count):
            merge_tmp.theta[tuple(p)] = c
        merge_tmp.theta /= merge_tmp.n
        merge_tmp.entropy = stats.entropy(merge_tmp.theta.flatten(), base=c_max)
        merge_tmp.mc = np.log(merge_tmp.n) / np.log(c_max) * np.power(2, len(merge_tmp))
        merge_tmp.cpc = merge_tmp.n * merge_tmp.entropy
        merge_tmp.cc = merge_tmp.mc + merge_tmp.cpc
        return merge_tmp

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
