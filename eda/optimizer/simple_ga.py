import numpy as np

from optimizer.eda_base import EDABase


class SimpleGA(EDABase):
    def __init__(self, categories, replacement, lam=16, theta_init=None,
                 selection=None, crossover=None, mutation=None,
                 crossover_prob=0.7, elite=False):
        super(SimpleGA, self).__init__(categories, lam=lam, theta_init=theta_init)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement
        self.crossover_prob = crossover_prob
        self.elite = elite

        self.counter = 0
        self.population = None
        self.fitness = None

    def update(self, c_one, fxc, range_restriction=False):
        self.counter = 0
        if self.population is None:
            self.eval_count += c_one.shape[0]
        else:
            self.eval_count += int(self.crossover_prob * c_one.shape[0])

        if self.elite and self.population is not None:
            self.eval_count -= 1
            worst_idx = np.argmax(fxc)
            best_idx = np.argmin(self.fitness)
            c_one[worst_idx] = self.population[best_idx]
            fxc[worst_idx] = self.fitness[best_idx]
        self.population = c_one
        self.fitness = fxc

        # store best individual and evaluation value
        best_idx = np.argmin(self.fitness)
        if self.best_eval > self.fitness[best_idx]:
            self.best_eval = self.fitness[best_idx]
            self.best_indiv = self.population[best_idx]
        # selection
        if self.selection is not None:
            self.population, self.fitness = self.selection(self.population,
                                                           self.fitness)
        # crossover
        if self.crossover is not None:
            p = self.fitness / np.sum(self.fitness)
            lam = int(self.crossover_prob * self.population.shape[0])
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
                self.fitness[idx1] = np.inf
                self.population[idx2] = child2
                self.fitness[idx2] = np.inf
        # mutation
        if self.mutation is not None:
            self.population = self.mutation(self.population)

    def sampling(self):
        if self.population is None:
            rand = np.random.rand(self.d, 1)
            cum_theta = self.theta.cumsum(axis=1)
            c = (cum_theta - self.theta <= rand) & (rand < cum_theta)
            return c
        else:
            indiv = self.population[self.counter]
            self.counter += 1
            return indiv

    def is_convergence(self):
        if self.population is None:
            return False
        else:
            return np.all(self.population == self.population[0])

    def __str__(self):
        sup_str = "    " + super(SimpleGA, self).__str__().replace("\n", "\n    ")
        sel_str = "    " + str(self.selection).replace("\n", "\n    ")
        cross_str = "    " + str(self.crossover).replace("\n", "\n    ")
        mut_str = "    " + str(self.mutation).replace("\n", "\n    ")
        return 'Simple GA(\n' \
               '{}\n' \
               '    elite: {}\n' \
               '    crossover prob: {}\n' \
               '{}\n' \
               '{}\n' \
               '{}\n' \
               ')'.format(sup_str, self.elite, self.crossover_prob,
                          sel_str, cross_str, mut_str)
