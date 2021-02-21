import numpy as np

from eda.optimizer.replacement import ReplacementBase


class Truncation(ReplacementBase):
    """
    A class of truncation replacement.
    """
    def __init__(self, replace_rate=0.5, fix_size=True):
        super(Truncation, self).__init__(replace_rate, fix_size=fix_size)

    def apply(self, parent, p_evals, candidate, c_evals):
        # sort parent by the evaluation value
        sorted_idx = np.argsort(p_evals)
        parent = parent[sorted_idx]
        p_evals = p_evals[sorted_idx]
        # replace individuals in parent with ones in candidate
        p_lam = parent.shape[0]
        c_lam = candidate.shape[0]
        replaced_lam = int(self.replace_rate * p_lam)
        if self.fix_size:
            assert replaced_lam == c_lam, \
                "The number of individuals for the replacement({}) must match the population size of candidate({})".format(replaced_lam, c_lam)
            parent[-c_lam:] = candidate
            p_evals[-c_lam:] = c_evals
        else:
            survived_lam = p_lam - replaced_lam
            parent = np.concatenate([parent[:survived_lam], candidate], axis=0)
            p_evals = np.concatenate([p_evals[:survived_lam], c_evals], axis=0)
        return parent, p_evals

    def __str__(self):
        sup_str = "    " + super(Truncation, self).__str__().replace("\n", "\n    ")
        return 'Truncation Replacement(\n' \
               '{}' \
               '\n)'.format(sup_str)
