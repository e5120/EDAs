import unittest
from unittest import TestCase, main

import numpy as np

from eda.optimizer.selection import Tournament


class TestTournament(TestCase):
    def test_selection(self):
        lam = 10
        num = 1000
        width = 100
        print("population size: {}\trange of input: 1-{}".format(lam, width))
        print("Average ranking of individuals chosen by tournament selection.")
        population = np.random.choice(np.arange(1, width), lam, replace=False)
        evals = population
        ranking = np.arange(1, lam+1)[np.argsort(np.argsort(evals))]

        rate, k = 0.2, 2
        selection = Tournament(rate, k=k)
        mean = 0
        for i in range(num):
            p, _ = selection(ranking, evals)
            mean += np.mean(p) / num
        print("k=2: {:.3f}".format(mean))

        k = 3
        selection = Tournament(rate, k=k)
        mean = 0
        for i in range(num):
            p, _ = selection(ranking, evals)
            mean += np.mean(p) / num
        print("k=3: {:.3f}".format(mean))

        k = 4
        selection = Tournament(rate, k=k)
        mean = 0
        for i in range(num):
            p, _ = selection(ranking, evals)
            mean += np.mean(p) / num
        print("k=4: {:.3f}".format(mean))

        k = 5
        selection = Tournament(rate, k=k)
        mean = 0
        for i in range(num):
            p, _ = selection(ranking, evals)
            mean += np.mean(p) / num
        print("k=5: {:.3f}".format(mean))

    def test_assert(self):
        with self.assertRaises(AssertionError):
            self.assertRaises(Tournament(0.5, 3))


if __name__ == "__main__":
    unittest.main()
