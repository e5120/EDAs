import numpy as np

from eda.optimizer.eda_base import EDABase
from eda.optimizer.metric import *
from eda.utils import estimate_cpt


class BOA(EDABase):
    """
    A class of bayesian optimization algorithm (BOA)
    """
    def __init__(self, categories, selection, replacement,
                 lam=500, k=None, criterion="bic", theta_init=None):
        """
        Parameters
        ----------
        selection : eda.optimizer.selection.selection_base.SelectionBase, default None
            Selection method.
        replacement : eda.optimizer.replacement.replacement_base.ReplacementBase
            Replacement method.
        k : int, default None
            Maximum number of parents of each node.
            If None, there is not constraint.
        criterion : string, default "bic"
            Evaluation criterion to measure the degree of dependencies among variables.
        """
        super(BOA, self).__init__(categories, lam=lam, theta_init=theta_init)
        assert k is None or 0 < k <= self.d
        assert criterion in ["aic", "bic", "k2"], \
            "Evaluation criterion takes one of three criterions, aic, bic, and k2."
        self.selection = selection
        self.replacement = replacement
        self.constraint_k = k
        if criterion == "aic":
            self.metric = AIC
        elif criterion == "bic":
            self.metric = BIC
        else:
            self.metric = K2

        self.population = None
        self.evaluation = None
        self.structure_mat = None
        self.sampling_order = None
        self.prob_mat = None

    def update(self, x, evals, range_restriction=False):
        x, evals = self._preprocess(x, evals)
        if self.population is None:
            self.population = x
            self.evaluation = evals
            self.lam = int(self.lam * self.replacement.replace_rate)
        else:
            self.population, self.evaluation = self.replacement(self.population,
                                                                self.evaluation,
                                                                x,
                                                                evals)
        x, evals = self.selection(self.population, self.evaluation)

        score = self.k2_algorithm(np.argmax(x, axis=2))

        info = {"structure": self.structure_mat,
                "population": np.argmax(self.population, axis=-1),
                "evaluations": self.evaluation,
                "probability": self.prob_mat,
                "k2_algorithm_score": score}
        return info

    def k2_algorithm(self, x):
        """
        K2 algorithm in order to build a Bayesian network.

        Parameters
        ----------
        x : numpy.ndarray
            Population.

        Returns
        -------
        float
            Evaluation value.
        """
        metric = self.metric(x, self.C)
        self.structure_mat = np.zeros((self.d, self.d))
        inheritance_mat = np.zeros((self.d, self.d))
        node_scores = np.zeros(self.d)
        child_counter = np.zeros(self.d)
        for node_idx in range(self.d):
            parents_idx = self.structure_mat[node_idx] != 0
            node_scores[node_idx] = metric.local_score(node_idx, parents_idx)

        min_gain = 0
        candidate_gains = np.zeros((self.d, self.d)) + min_gain
        for i in range(self.d):
            for j in range(i+1, self.d):
                parents_idx = self.structure_mat[i] != 0
                parents_idx[j] = True
                new_score = metric.local_score(i, parents_idx)
                candidate_gains[i, j] = new_score - node_scores[i]
                candidate_gains[j, i] = candidate_gains[i, j]
        # build graph structure with greedy algorithm
        while self.constraint_k is None or self.constraint_k > 0:
            child, parent = np.unravel_index(np.argmax(candidate_gains), candidate_gains.shape)
            if candidate_gains[child, parent] == min_gain:
                break
            self.structure_mat[child, parent] = 1
            node_scores[child] += candidate_gains[child, parent]

            candidate_gains[child, parent] = min_gain
            candidate_gains[parent, child] = min_gain

            inheritance_mat[child, parent] = 1

            new_ancestors = [parent]

            for i in range(self.d):
                if inheritance_mat[parent, i] == 1 and inheritance_mat[child, i] == 0:
                    new_ancestors.append(i)
                    inheritance_mat[child, i] = 1
                    candidate_gains[i, child] = min_gain

            descendents = np.where(inheritance_mat[:, child] == 1)[0].tolist()
            while len(descendents) > 0:
                node = descendents.pop(0)

                node_updated = False
                for ancestor in new_ancestors:
                    if inheritance_mat[node, ancestor] == 0:
                        inheritance_mat[node, ancestor] = 1
                        candidate_gains[ancestor, node] = min_gain
                        node_updated = True
                if node_updated:
                    descendents.extend(np.where(inheritance_mat[:, node] == 1)[0])
                    descendents = list(set(descendents))
            # update candidate gains for affected child node
            for i in range(self.d):
                if self.structure_mat[child, i] == 0 and inheritance_mat[i, child] == 0 and child != i:
                    parents_idx = self.structure_mat[child] != 0
                    parents_idx[i] = True
                    new_score = metric.local_score(child, parents_idx)
                    candidate_gains[child, i] = new_score - node_scores[child]

            child_counter[child] += 1
            if self.constraint_k is not None and child_counter[child] == self.constraint_k:
                candidate_gains[child] = min_gain

        self.prob_mat = []
        for i in range(self.d):
            parents_idx = self.structure_mat[i] == 1
            cpt = estimate_cpt(x[:, i], x[:, parents_idx], base=self.Cmax)
            self.prob_mat.append(cpt)
        # deceide sampling order of each node
        tmp_structure = self.structure_mat.copy()
        s = list(np.where(np.sum(tmp_structure, axis=1) == 0)[0])
        self.sampling_order = []
        while len(s) > 0:
            n = s.pop(0)
            self.sampling_order.append(n)

            candidates = np.where(tmp_structure[:, n] == 1)[0]
            for i in range(len(candidates)):
                m = candidates[i]
                tmp_structure[m, n] = 0
                if np.all(tmp_structure[m] == 0):
                    s.append(m)
        if np.any(tmp_structure != 0):
            print("error. graph has cycles")
            exit(-1)

        return metric.score(self.structure_mat)

    def sampling(self):
        # random sampling, only first generation
        if self.prob_mat is None:
            return super(BOA, self).sampling()
        # sample by using each probability of bayesian network
        else:
            c = np.zeros((self.d, self.Cmax), dtype=bool)
            for i in self.sampling_order:
                rand = np.random.rand()
                parents = self.structure_mat[i] == 1
                val = np.argmax(c[parents], axis=1)
                if len(val) == 0:
                    theta = self.prob_mat[i]
                else:
                    theta = self.prob_mat[i][tuple(val)]
                    if theta.shape[-1] == 0:
                        theta = theta.squeeze(-1)
                cum_theta = theta.cumsum()
                _c = (cum_theta - theta <= rand) & (rand < cum_theta)
                c[i, _c] = True
            return c

    def convergence(self):
        if self.sampling_order is None:
            return 0.5
        convergence = 0.0
        dummy = np.zeros(self.d, dtype=np.int)
        for c in self.sampling_order:
            p = self.structure_mat[c] == 1
            prob = self.prob_mat[c][tuple(dummy[p])]
            dummy[c] = np.argmax(prob)
            convergence += np.max(prob)
        convergence /= self.d
        return convergence

    def __str__(self):
        sup_str = "    " + super(BOA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        rep_str = "    " + str(self.replacement).replace("\n", "\n    ")
        return 'BOA(\n' \
               '{}\n' \
               '    maximum number of parents: {}\n' \
               '    evaluation criterion: {}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, self.constraint_k, self.metric.__name__, sel_str, rep_str)
