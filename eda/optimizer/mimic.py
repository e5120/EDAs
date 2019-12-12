import numpy as np
from scipy import stats

from optimizer.eda_base import EDABase


class MIMIC(EDABase):
    def __init__(self, categories, replacement,
                 replace_rate=0.2, lam=128, theta_init=None):
        super(MIMIC, self).__init__(categories, lam=lam, theta_init=theta_init)
        self.replace_rate = replace_rate
        self.replacement = replacement
        self.total_lam = self.lam
        self.population = None
        self.fitness = None
        self.uni_freq = None
        self.bi_freq = None
        self.order = None

    def update(self, c_one, fxc, range_restriction=False):
        self.eval_count += c_one.shape[0]
        # sort by fitness and get the index of the top of "selection_rate"%
        best_idx = np.argmin(fxc)
        # store best individual and evaluation value
        if self.best_eval > fxc[best_idx]:
            self.best_eval = fxc[best_idx]
            self.best_indiv = c_one[best_idx]
        # replacement
        if self.population is None:
            self.population = c_one
            self.fitness = fxc
            self.lam = int(np.ceil(self.lam * self.replace_rate))
        else:
            self.population, self.fitness = self.replacement(self.population,
                                                             self.fitness,
                                                             c_one,
                                                             fxc)
        # get parent population
        idx = np.argsort(self.fitness)
        self.population = self.population[idx]
        self.fitness = self.fitness[idx]
        parent = self.population[:self.total_lam-self.lam]
        # get frequency of each bit
        self.uni_freq = self.calc_uni_frequency(parent)
        # if range_restriction is True, clipping
        for i in range(self.d):
            ci = self.C[i]
            theta_min = 1.0 / (self.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.0
            self.theta[i, :ci] = np.maximum(self.theta[i, :ci], theta_min)
            theta_sum = self.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta[i, :ci] -= (theta_sum - 1.0) * (self.theta[i, :ci] - theta_min) / tmp
            self.theta[i, :ci] /= self.theta[i, :ci].sum()
        # get frequency of at each pairwise bits
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
        if self.total_lam == self.lam:
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

    def __str__(self):
        sup_str = "    " + super(MIMIC, self).__str__().replace("\n", "\n    ")
        return 'MIMIC(\n' \
               '{}\n' \
               '    replace rate: {}\n' \
               ')'.format(sup_str, self.replace_rate)
