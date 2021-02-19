import numpy as np

from eda.optimizer.eda_base import EDABase
from eda.optimizer.util import SubSet, Cache


class ECGA(EDABase):
    """
    A class of extended compact genetic algorithm (ECGA).
    """
    def __init__(self, categories, replacement,
                 selection=None, lam=500, theta_init=None):
        """
        Parameters
        ----------
        replacement : eda.optimizer.replacement.replacement_base.ReplacementBase
            Replacement method.
        selection : eda.optimizer.selection.selection_base.SelectionBase, default None
            Selection method.
        """
        super(ECGA, self).__init__(categories, lam=lam, theta_init=theta_init)
        self.replacement = replacement
        self.selection = selection

        self.population = None
        self.fitness = None
        self.cluster = None

    def update(self, x, evals, range_restriction=False):
        x, evals = self._preprocess(x, evals)
        if self.selection is not None:
            x, evals = self.selection(x, evals)
        x = np.argmax(x, axis=2)
        if self.population is None:
            self.population = x
            self.fitness = evals
            self.lam = int(self.lam * self.replacement.replace_rate)
        else:
            self.population, self.fitness = self.replacement(self.population,
                                                             self.fitness,
                                                             x,
                                                             evals)
        self.cluster = self.greedy_mpm_search(self.population)

    def greedy_mpm_search(self, population):
        """
        Build a greedy magrinal marginal product model.

        Parameters
        ----------
        population : numpy.ndarray
            Population.

        Returns
        -------
        Variables after clustering.
        """
        # initialize subset of cluster
        cluster = [SubSet(i, population[:, i], self.Cmax) for i in range(self.d)]
        # initialize cache
        cache = self.initialize_mpm(cluster)
        # clustering according to CCO
        while True:
            pos_i, pos_j = cache.argmax_cc()
            if cache.cc_list[pos_i, pos_j] <= 0:
                break
            cluster[pos_i] = cache.subsets[pos_i, pos_j]
            cluster.pop(pos_j)
            cache.remove(pos_j)
            for k in range(len(cluster)):
                if k == pos_i:
                    continue
                i, k = (k, pos_i) if k < pos_i else (pos_i, k)
                merge = cluster[i].merge(cluster[k])
                cache.add(i, k, cluster[i].cc + cluster[k].cc - merge.cc, merge)
        return cluster

    def initialize_mpm(self, cluster):
        """
        Initialize a marginal product model.

        Parameters
        ----------
        cluster : list
            cluster group.

        Returns
        -------
        eda.optimizer.util.cache.Cache
            Cache object for fast computation.
        """
        cache = Cache(self.d)
        for i in range(len(cluster)-1):
            for j in range(i+1, len(cluster)):
                subset1 = cluster[i]
                subset2 = cluster[j]
                merge = subset1.merge(subset2)
                cache.add(i, j, subset1.cc + subset2.cc - merge.cc, merge)
        return cache

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

    def convergence(self):
        if self.cluster is None:
            return 0.5
        return np.mean([np.max(c.theta) for c in self.cluster])

    def __str__(self):
        sup_str = "    " + super(ECGA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        rep_str = "    " + str(self.replacement).replace("\n", "\n    ")
        return 'ECGA(\n' \
               '{}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, sel_str, rep_str)
