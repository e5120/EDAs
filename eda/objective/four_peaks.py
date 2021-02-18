import numpy as np

from eda.objective import ObjectiveBase


class FourPeaks(ObjectiveBase):
    """
    A class of four-peaks function.
    f(c) = max(o(c), z(c)) + REWARD,
    o(c) = The number of contiguous ones starting in position 1,
    z(c) = The number of contiguous zeros ending in position D,
    REWARD = D if o(c) > T and z(c) > T else 0,
    where c = (c_1, c_2, ..., c_D) and T is a user parameter.

    Reference:
    https://www.ri.cmu.edu/pub_files/pub2/baluja_shumeet_1995_1/baluja_shumeet_1995_1.pdf
    """
    def __init__(self, dim, t, minimize=True):
        super(FourPeaks, self).__init__(dim, minimize=minimize)
        assert 0 < t < dim // 2
        self.t = t

        optimal_value = 2 * dim - t - 1
        self.optimal_value = -optimal_value if minimize else optimal_value
        self.optimal_indiv = np.zeros(self.d, dtype=np.int)
        self.optimal_indiv[:self.t+1] = 1

    def evaluate(self, c):
        c = self._check_shape(c)
        c_inv = c[:, ::-1]
        o_c = np.argmin(c, axis=1)
        z_c = np.argmax(c_inv, axis=1)
        evals = np.maximum(o_c, z_c)
        evals = evals + np.where(np.logical_and(o_c > self.t, z_c > self.t),
                                 self.d,
                                 0)
        evals = -evals if self.minimize else evals
        info = {"o_c": o_c, "z_c": z_c}
        return evals, info

    def __str__(self):
        sup_str = "    " + super(FourPeaks, self).__str__().replace("\n", "\n    ")
        return 'Foue-Peaks(\n' \
               '{}\n' \
               '    Threshold T: {}\n' \
               ')\n'.format(sup_str, self.t)
