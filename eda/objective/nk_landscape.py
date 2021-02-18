import numpy as np

from eda.objective import ObjectiveBase
from eda.utils import packbits


class NKLandscape(ObjectiveBase):
    """
    A class of NK-landscape.
    The NK-landscape has a parameter k which specifies that how many bits each bit depends on.
    The evaluation value of a bit is obtained from a lookup table by using a combination of values of k+1 bits and the evaluation value of an individual is calculated by averaging the evaluation values of all bits.
    Here, the k bits and the lookup table are randomly generated in advance.

    Reference:
    http://ncra.ucd.ie/wp-content/uploads/2020/08/SocialLearning_GECCO2019.pdf
    """
    def __init__(self, dim, k=2, seed=1, minimize=True):
        """
        Parameters
        ----------
        k : int, default 2
            A user parameter which is determined the ruggedness of the problem.
            How many bits each bit depends on.
        seed : int, default 1
            A random number seed.
            The seed is required to determine the k bits and the lookup table.
        """
        super(NKLandscape, self).__init__(dim, minimize=minimize)
        assert 0 < k < dim
        self.k = k
        if seed is not None:
            next_seed = np.random.randint(0, 2**31)
            np.random.seed(seed)
        self.eval_table = self.get_evaluation_table()
        self.neighbor = self.get_neighbor()
        if seed is not None:
            np.random.seed(next_seed)

    def evaluate(self, c):
        c = self._check_shape(c)
        evals = np.zeros(c.shape[0])
        for d in range(self.dim):
            bits = np.concatenate([c[:, d][:, np.newaxis], c[:, self.neighbor[d]]], axis=1)
            evals += self.eval_table[d][packbits(bits)]
        evals /= self.dim
        evals = -evals if self.minimize else evals
        info = {}
        return evals, info

    def get_evaluation_table(self):
        """
        Generate a lookup table.
        """
        return np.array([
                np.random.choice(np.arange(2**(self.k+1)), 2**(self.k+1), replace=False) / (2**(self.k+1) - 1)
                for _ in range(self.dim)
            ])

    def get_neighbor(self):
        """
        Determine the k bits which depends on each bit.
        """
        neighbor = np.zeros((self.dim, self.dim), dtype=np.bool)
        for i in range(self.dim):
            _k = 0
            while _k < self.k:
                n = np.random.choice(self.dim)
                if i == n or neighbor[i, n]:
                    continue
                neighbor[i, n] = True
                _k += 1
        return neighbor
