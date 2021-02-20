import numpy as np

from eda.objective import ObjectiveBase


class WModel(ObjectiveBase):
    """
    A class of w-model function.

    The w-model function is a benchmark function which combines several properties of the black-box discrete optimization.
    The properties introduced into w-model function is as follows.
    1)Neutrality
    2)Epistasis
    3)Multi-objectivity
    4)Ruggedness and deceptiveness

    Reference:
    http://iao.hfuu.edu.cn/images/publications/W2018TWMATBBDOBPIFTBGW.pdf
    """
    def __init__(self, dim, mu=2, v=4, m=2, n=5, gamma=0, minimize=True):
        """
        Parameters
        ----------
        mu : int, default 2
            A user parameter for neutrality.
        v : int, default 4
            A user parameter for epistasis.
        m : int, default 2
            A user parameter for multi-objectivity.
        n : int, default 5
            The dimension of each objective function.
        gamma : int, default 0
            A user parameter for ruggedness and deceptiveness.
        """
        super(WModel, self).__init__(dim, minimize=minimize)
        assert 0 < mu <= dim
        assert 1 < v <= dim
        assert 0 < m
        assert 0 < n < dim
        assert 0 <= gamma <= int(n * (n - 1) / 2)
        self.mu = mu
        self.v = v
        self.m = m
        self.n = n
        self.gamma = gamma
        self.perm = self._permutate(self._translate(gamma, n), n)
        self.target = np.zeros(n, dtype=np.int)
        self.target[0::2] = 1
        self.optimal_value = 0.0

    def evaluate(self, c):
        c = self._check_shape(c)
        c = self.neutrality_layer(c)
        c = self.epistasis_layer(c)
        c = self.multi_objective_layer(c)
        evals = self.evaluation_layer(c)
        evals = self.ruggedness_and_deceptiveness_layer(evals)
        evals = np.sum(evals, axis=-1)
        evals = evals if self.minimize else -evals
        info = {}
        return evals, info

    def neutrality_layer(self, c):
        """
        Neutrality is introduced into w-model function.
        """
        if self.dim % self.mu != 0:
            c = c[:, :-(self.dim % self.mu)]
        c = c.reshape(c.shape[0], -1, self.mu)
        c = np.sum(c, axis=-1)
        c = np.where(c < self.mu / 2, 0, 1)
        return c

    def epistasis_layer(self, c):
        """
        Epistasis is introduced into w-model function.
        """
        dim = c.shape[1]
        if dim % self.v != 0:
            remain = c[:, -(dim % self.v):]
            c = c[:, :-(dim % self.v)]
        c = self._epistasis(c, self.v)
        if dim % self.v != 0:
            remain = self._epistasis(remain, dim % self.v)
            c = np.c_[c, remain]
        return c

    def _epistasis(self, x, v):
        x = x.reshape(x.shape[0], -1, v)
        th = np.where(x[:, :, 0] == 1)
        x[:, :, 0] = 0
        x = np.roll(x, -1, axis=-1)
        x = np.where((np.sum(x, axis=-1)[:, :, np.newaxis] - x) % 2, 1, 0)
        x[th] = np.logical_not(x[th])
        x = x.reshape(x.shape[0], -1)
        return x

    def multi_objective_layer(self, c):
        """
        Multi-objectivity is introduced into w-model function.
        """
        y = np.tile(self.target, (c.shape[0], self.m, 1))
        for i in range(self.m):
            target_string = c[:, i::self.m]
            if target_string.shape[1] == self.n:
                y[:, i] = target_string
            elif target_string.shape[1] < self.n:
                y[:, i, :target_string.shape[1]] = target_string
            elif target_string.shape[1] > self.n:
                y[:, i] = target_string[:, :self.n]
        return y

    def evaluation_layer(self, c):
        """
        Calculate the evaluation value of each objective function.
        """
        return np.sum(c != self.target, axis=-1)

    def ruggedness_and_deceptiveness_layer(self, c):
        """
        Ruggedness and deceptiveness are introduced into w-model function.
        """
        return self.perm[c]

    def _translate(self, gamma, n):
        if gamma <= 0:
            return 0
        l = int(n * (n - 1) / 2)
        i = int(n / 2) * int((n + 1) / 2)
        if gamma <= i:
            j = int((n + 2) / 2 - np.sqrt(n**2 / 4 + 1 - gamma))
            k = gamma - j * (n + 2) + j**2 + n
            return k + 2 * (j * (n + 2) - j**2 - n) - j
        else:
            j = int((n % 2 + 1) / 2 + np.sqrt((1 - n % 2) / 4 + gamma - 1 -i))
            k = gamma - (j - n % 2) * (j - 1) - 1 - i
            return l - k - 2 * j**2 + j - (n % 2) * (-2 * j + 1)

    def _permutate(self, gamma, n):
        perm = np.zeros(n + 1, dtype=np.int)
        upper = int(n * (n - 1) / 2)
        start = 0 if gamma <= 0 else n - 1 - int(0.5 + np.sqrt(0.25 + 2 * (upper - gamma)))
        j, k = 0, 0
        for j in range(1, start + 1):
            if j % 2 == 1:
                perm[j] = n - k
            else:
                k += 1
                perm[j] = k
        for j in range(j + 1, n + 1):
            k += 1
            perm[j] = n - k if start % 2 == 1 else k
        upper = int(gamma - upper + (n - start  - 1) * (n - start) / 2)
        j = n
        for _ in range(1, upper + 1):
            j -= 1
            perm[j], perm[n] = perm[n], perm[j]
        return perm

    def __str__(self):
        sup_str = "    " + super(WModel, self).__str__().replace("\n", "\n    ")
        return 'W-Model(\n' \
               '{}\n' \
               '    mu: {}\n' \
               '    v: {}\n' \
               '    m: {}\n' \
               '    n: {}\n' \
               '    gamma: {}\n' \
               ')\n'.format(sup_str, self.mu, self.v,
                            self.m, self.n, self.gamma)
