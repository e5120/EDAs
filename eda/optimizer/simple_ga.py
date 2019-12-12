import numpy as np

from optimizer.eda_base import EDABase


class SimpleGA(EDABase):
    def __init__(self, categories, lam=16, theta_init=None,
                 selection=None, crossover=None, mutation=None,
                 crossover_prob=0.7, mutation_prob=0.01):
        super(SimpleGA, self).__init__(categories, lam=lam, theta_init=theta_init)
        assert self.Cmax == 2
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = None
        self.fitness = None
        self.next_gen_idxes = []
        self.counter = 0

    def update(self, c_one, fxc, range_restriction=False):
        self.counter = 0
        if self.population is None:
            self.population = c_one
            self.fitness = fxc
            self.eval_count += c_one.shape[0]
        else:
            self.fitness[self.next_gen_idxes] = fxc[: len(self.next_gen_idxes)]
            self.eval_count += len(self.next_gen_idxes)
        # store best individual and evaluation value
        idx = np.argsort(self.fitness)
        if self.best_eval > self.fitness[idx[0]]:
            self.best_eval = self.fitness[idx[0]]
            self.best_indiv = self.population[idx[0]]
        # selection
        if self.selection is not None:
            self.population, self.fitness = self.selection(self.population, self.fitness, sort=False)
        # crossover
        if self.crossover is not None:
            p = self.fitness / np.sum(self.fitness)
            lam = int(self.crossover_prob * self.lam)
            lam = lam if lam % 2 == 0 else lam - 1
            parents_idx = np.random.choice(np.arange(self.lam), lam, replace=False, p=p)
            for i in range(0, len(parents_idx), 2):
                idx1 = parents_idx[i]
                idx2 = parents_idx[i+1]
                child1, child2 = self.crossover(self.population[idx1],
                                                self.population[idx2],
                                                self.fitness[idx1],
                                                self.fitness[idx2])
                self.population[idx1] = child1
                self.population[idx2] = child2
            self.next_gen_idxes = parents_idx
        # mutation
        if self.mutation is not None:
            pass

    def sampling(self):
        if len(self.next_gen_idxes) > self.counter:
            self.counter += 1
            return self.population[self.next_gen_idxes[self.counter - 1]]

        rand = np.random.rand(self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)
        c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return c

    def __str__(self):
        sup_str = "    " + super(SimpleGA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        cross_str = "    " + str(self.crossover).replace("\n", "\n    ")
        mut_str = "    " + str(self.mutation).replace("\n", "\n    ")
        return 'GA(\n' \
               '{}\n' \
               '    crossover prob: {}\n' \
               '    mutation prob: {}\n' \
               '{}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, self.crossover_prob, self.mutation_prob,
                          sel_str, cross_str, mut_str)
