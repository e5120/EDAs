import numpy as np

from optimizer.eda_base import EDABase
from optimizer.util import SubSet, Cache


class ECGA(EDABase):
    def __init__(self, categories, replacement,
                 selection=None, lam=16, theta_init=None):
        super(ECGA, self).__init__(categories, lam=lam, theta_init=theta_init)
        self.cur_pop = None
        self.cur_fit = None
        self.selection = selection
        self.replacement = replacement
        self.cluster = None
        self.cache = Cache(self.d)

    def update(self, c_one, fxc, range_restriction=False):
        self.eval_count += c_one.shape[0]
        # store best individual and evaluation value
        best_idx = np.argmin(fxc)
        if self.best_eval > fxc[best_idx]:
            self.best_eval = fxc[best_idx]
            self.best_indiv = c_one[best_idx]
        if self.selection is not None:
            c_one, fxc = self.selection(c_one, fxc)
        c_one = np.argmax(c_one, axis=2)
        if self.cur_pop is None:
            self.cur_pop = c_one
            self.cur_fit = fxc
        else:
            self.cur_pop, self.cur_fit = self.replacement(self.cur_pop,
                                                          self.cur_fit,
                                                          c_one,
                                                          fxc)
        self.cluster = self.greedy_mpm_search(self.cur_pop)

    def greedy_mpm_search(self, population):
        # initialize subset of cluster
        cluster = [SubSet(i, population[:, i], self.Cmax) for i in range(self.d)]
        # initialize cache
        self.initialize_mpm(cluster)
        # clustering according to CCO
        while True:
            best_pos_i, best_pos_j = self.cache.argmax_cc()
            if self.cache.cc_list[best_pos_i, best_pos_j] <= 0:
                break
            cluster[best_pos_i] = self.cache.subsets[best_pos_i, best_pos_j]
            cluster.pop(best_pos_j)
            self.cache.remove(best_pos_j)
            for k in range(len(cluster)):
                if k == best_pos_i:
                    continue
                i, k = (k, best_pos_i) if k < best_pos_i else (best_pos_i, k)
                merge = cluster[i].merge(cluster[k], self.Cmax)
                self.cache.add(i, k, cluster[i].cc + cluster[k].cc - merge.cc, merge)
        return cluster

    def initialize_mpm(self, cluster):
        self.cache.init()
        for i in range(len(cluster)-1):
            for j in range(i+1, len(cluster)):
                subset1 = cluster[i]
                subset2 = cluster[j]
                merge = subset1.merge(subset2, self.Cmax)
                self.cache.add(i, j, subset1.cc + subset2.cc - merge.cc, merge)

    def sampling(self):
        # random sampling, only first generation
        if self.cluster is None:
            rand = np.random.rand(self.d, 1)
            cum_theta = self.theta.cumsum(axis=1)

            c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
            return c
        # sample by using each probability of cluster that clustered by CCO
        else:
            c = np.zeros((self.d, self.Cmax), dtype=bool)
            for cl in self.cluster:
                rand = np.random.rand()
                cum_theta = cl.theta.cumsum().reshape(cl.theta.shape)
                _c = (cum_theta - cl.theta <= rand) & (rand < cum_theta)
                if len(cl) > 1:
                    _c = np.unravel_index(np.argmax(_c), _c.shape)
                c[cl.idx_set, _c] = True
            return c

    def get_c_m(self, cluster):
        return np.sum([cl.mc for cl in cluster])

    def get_c_p(self, cluster):
        return np.sum([cl.cpc for cl in cluster])

    def get_c(self, cluster):
        return np.sum([cl.cc for cl in cluster])

    def is_convergence(self):
        if self.cluster is None:
            return False
        return np.abs(np.mean([np.max(c.theta) for c in self.cluster]) - 1.0) < 1e-8

    def __str__(self):
        sup_str = "    " + super(ECGA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        rep_str = "    " + str(self.replacement).replace("\n", "\n    ")
        return 'ECGA(\n' \
               '{}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, sel_str, rep_str)
