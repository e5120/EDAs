import numpy as np


class EDABase(object):
    def __init__(self, categories, lam=2, theta_init=None):
        self.N = np.sum(categories - 1)

        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1.0 / self.C[i]
        self.valid_params = int(np.sum(self.C - 1))
        self.valid_d = len(self.C[self.C > 1])

        if theta_init is not None:
            self.theta = theta_init

        self.lam = lam
        # for record
        self.best_eval = np.inf
        self.best_indiv = None
        self.eval_count = 0

    def sampling(self):
        rand = np.random.rand(self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)

        c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return c

    def is_convergence(self, eps=1e-8):
        return np.abs(self.theta.max(axis=1).mean() - 1.0) < eps

    def update(self, c_one, fxc, range_restriction=False):
        raise NotImplementedError

    def __str__(self):
        return 'param size: {}\n' \
               'population size: {}\n' \
               'max number of choices: {}'.format(self.valid_params, self.lam, self.Cmax)
