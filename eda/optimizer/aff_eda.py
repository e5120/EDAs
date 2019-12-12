import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn import cluster

from optimizer.eda_base import EDABase
from optimizer.util import SubSet


class AffEDA(EDABase):
    def __init__(self, categories, replacement,
                 selection=None, lam=16, theta_init=None):
        super(AffEDA, self).__init__(categories, lam=lam, theta_init=theta_init)
        assert self.Cmax == 2
        self.cur_pop = None
        self.cur_fit = None
        self.selection = selection
        self.replacement = replacement
        self.ap = cluster.AffinityPropagation(affinity="precomputed")
        self.cluster = None

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
        mi = self.calc_mutual_information(self.cur_pop)
        cluster_label = self.ap.fit(mi).labels_
        # print(cluster_label)
        # print(self.ap.n_iter_)
        cluster_num = np.max(cluster_label) + 1
        cluster = [None for _ in range(cluster_num)]
        for i, label in enumerate(cluster_label):
            subset = SubSet(i, self.cur_pop[:, i], self.Cmax)
            if cluster[label] is None:
                cluster[label] = subset
            else:
                cluster[label] = cluster[label].merge(subset, self.Cmax)
        self.cluster = cluster

    def calc_mutual_information(self, population):
        dim = population.shape[1]
        mi = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i, dim):
                mi[i, j] = mutual_info_score(population[:, i], population[:, j])
                mi[j, i] = mi[i, j]
        return mi

    def sampling(self):
        if self.cluster is None:
            rand = np.random.rand(self.d, 1)
            cum_theta = self.theta.cumsum(axis=1)

            c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
            return c
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

    def is_convergence(self):
        if self.cluster is None:
            return False
        return np.abs(np.mean([np.max(c.theta) for c in self.cluster]) - 1.0) < 1e-8

    def __str__(self):
        sup_str = "    " + super(AffEDA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        rep_str = "    " + str(self.replacement).replace("\n", "\n    ")
        return 'AffEDA(\n' \
               '{}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, sel_str, rep_str)
