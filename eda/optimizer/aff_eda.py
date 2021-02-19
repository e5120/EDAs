import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn import cluster

from eda.optimizer.eda_base import EDABase
from eda.optimizer.util import SubSet


class AffEDA(EDABase):
    """
    A class of affinity propagation estimation of distribution algorithm (AffEDA).
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
        super(AffEDA, self).__init__(categories, lam=lam, theta_init=theta_init)
        self.replacement = replacement
        self.selection = selection

        self.population = None
        self.fitness = None
        self.cluster = None
        self.ap = cluster.AffinityPropagation(affinity="precomputed", random_state=0)

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
        mi = self.calc_mutual_information(self.population)
        cluster_label = self.ap.fit(mi).labels_
        cluster_num = np.max(cluster_label) + 1
        cluster = [None for _ in range(cluster_num)]
        for i, label in enumerate(cluster_label):
            subset = SubSet(i, self.population[:, i], self.Cmax)
            cluster[label] = subset if cluster[label] is None else cluster[label].merge(subset)
        self.cluster = cluster

    def calc_mutual_information(self, population):
        """
        Estimate mutual information.

        Parameters
        ----------
        population : numpy.ndarray
            Population.

        Returns
        -------
        numpy.ndarray
            Mutual information estimated using the population.
        """
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

    def convergence(self):
        if self.cluster is None:
            return 0.5
        return np.mean([np.max(c.theta) for c in self.cluster])

    def __str__(self):
        sup_str = "    " + super(AffEDA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        rep_str = "    " + str(self.replacement).replace("\n", "\n    ")
        return 'AffEDA(\n' \
               '{}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, sel_str, rep_str)
