import numpy as np
from scipy import stats

from eda.optimizer.eda_base import EDABase


class MIMIC(EDABase):
    """
    A class of mutual-information-maximizing input clustering (MIMIC).
    """
    def __init__(self, categories, replacement, lam=128, theta_init=None):
        """
        Parameters
        ----------
        replacement : eda.optimizer.replacement.replacement_base.RepalcementBase
            Replacement method.
        """
        super(MIMIC, self).__init__(categories, lam=lam, theta_init=theta_init)
        self.replacement = replacement

        self.population = None
        self.fitness = None
        self.uni_freq = None
        self.bi_freq = None
        self.order = None

    def update(self, x, evals, range_restriction=False):
        x, evals = self._preprocess(x, evals)
        if self.population is None:
            self.population = x
            self.fitness = evals
            self.lam = int(np.ceil(self.lam * self.replacement.replace_rate))
        else:
            self.population, self.fitness = self.replacement(self.population,
                                                             self.fitness,
                                                             x,
                                                             evals)
        # get parent population
        idx = np.argsort(self.fitness)
        self.population = self.population[idx]
        self.fitness = self.fitness[idx]
        parent = self.population[:-self.lam]
        # get frequency of each bit
        self.uni_freq = self.calc_uni_frequency(parent)
        # get frequency of each pairwise bits
        self.bi_freq = self.calc_bi_frequency(parent)
        # get permutation of binary bit
        self.order = self.calc_permutation_order(self.uni_freq, self.bi_freq)

    def calc_uni_frequency(self, population):
        lam = population.shape[0]
        return np.sum(population, axis=0) / lam

    def calc_bi_frequency(self, population):
        population = np.argmax(population, axis=2)
        lam = population.shape[0]
        dim = population.shape[1]
        bi_freq = np.zeros((dim, dim, self.Cmax**2))
        for m in range(dim):
            for n in range(m, dim):
                counter = np.zeros((self.Cmax, self.Cmax))
                for j in range(lam):
                    x = population[j, m]
                    y = population[j, n]
                    counter[x, y] += 1
                bi_freq[m, n] = counter.flatten()
                bi_freq[n, m] = counter.T.flatten()
        bi_freq /= lam
        return bi_freq

    def calc_permutation_order(self, uni_freq, bi_freq):
        # decide first bit
        dim = uni_freq.shape[0]
        order = np.zeros(dim, dtype=int)
        used = np.zeros(dim)
        uni_entropy = stats.entropy(uni_freq.transpose())
        cur_bit = np.argmin(uni_entropy)
        order[0] = cur_bit
        used[cur_bit] = 1
        # decide following bit
        for m in range(1, dim):
            joint_entropy = np.where(used == 0,
                                     -np.sum(bi_freq[cur_bit] * self.safe_log(bi_freq[cur_bit]), axis=1),
                                     np.Infinity)
            cond_entropy = joint_entropy - uni_entropy[cur_bit]
            order[m] = np.argmin(cond_entropy)
            cur_bit = order[m]
            used[cur_bit] = 1
        return order

    def safe_log(self, vec):
        return np.where(vec > 0, np.log(vec), 0)

    def sampling(self):
        # only first generation
        if self.population is None:
            rand = np.random.rand(self.d, 1)
            cum_theta = self.theta.cumsum(axis=1)
            c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
            return c
        # otehrwise
        else:
            c = np.zeros((self.d, self.Cmax), dtype=bool)
            cur_bit = self.order[0]
            rand = np.random.rand()
            cum_theta = self.uni_freq[cur_bit].cumsum()
            c[cur_bit] = (cum_theta - self.uni_freq[cur_bit] <= rand) & (rand < cum_theta)

            for k in range(1, self.d):
                cur_bit_num = np.argmax(c[cur_bit])
                p1 = self.uni_freq[cur_bit, cur_bit_num]
                p12 = self.bi_freq[cur_bit, self.order[k], cur_bit_num*self.Cmax:(cur_bit_num+1)*self.Cmax]
                p2_1 = p12 / p1
                rand = np.random.rand()
                cum_theta = p2_1.cumsum()
                c[self.order[k]] = (cum_theta - p2_1 <= rand) & (rand < cum_theta)
                cur_bit = self.order[k]
            return c

    def convergence(self):
        # ToDo: implement
        return super().convergence()

    def __str__(self):
        sup_str = "    " + super(MIMIC, self).__str__().replace("\n", "\n    ")
        rep_str = self.replacement.__str__().replace("\n", "\n    ")
        return 'MIMIC(\n' \
               '{}\n' \
               '    {}\n' \
               ')'.format(sup_str, rep_str)
