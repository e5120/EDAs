import unittest
from unittest import TestCase, main

import numpy as np

from eda.optimizer.selection import Roulette


class TestRoulette(TestCase):
    def test_selection(self):
        rate = 0.5
        selection_eval = Roulette(rate, criterion="eval")
        selection_rank = Roulette(rate, criterion="rank")

        lam = 10
        num = 1000
        width = 100
        print("population size: {}\trange of input: 1-{}".format(lam, width))
        print("Average ranking of individuals chosen by roulette selection.")

        population = np.random.choice(np.arange(1, width), lam, replace=False)
        evals = population
        ranking = np.arange(1, lam+1)[np.argsort(np.argsort(evals))]
        mean_eval, mean_rank = 0, 0
        for i in range(num):
            p, _ = selection_eval(ranking, evals)
            mean_eval += np.mean(p) / num
            p, _ = selection_rank(ranking, evals)
            mean_rank += np.mean(p) / num
        print("Linear     : eval={:.3f}\trank={:.3f}".format(mean_eval, mean_rank))

        evals = np.exp(population)
        mean_eval, mean_rank = 0, 0
        for i in range(num):
            p, _ = selection_eval(ranking, evals)
            mean_eval += np.mean(p) / num
            p, _ = selection_rank(ranking, evals)
            mean_rank += np.mean(p) / num
        print("Exponential: eval={:.3f}\trank={:.3f}".format(mean_eval, mean_rank))

        evals = np.log(population)
        mean_eval, mean_rank = 0, 0
        for i in range(num):
            p, _ = selection_eval(ranking, evals)
            mean_eval += np.mean(p) / num
            p, _ = selection_rank(ranking, evals)
            mean_rank += np.mean(p) / num
        print("logarithm  : eval={:.3f}\trank={:.3f}".format(mean_eval, mean_rank))


if __name__ == "__main__":
    unittest.main()
