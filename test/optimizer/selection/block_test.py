import unittest
from unittest import TestCase, main

import numpy as np

from eda.optimizer.selection import Block


class TestTop(TestCase):
    def test_selection_1d(self):
        lam = 10
        population = np.arange(lam)
        evals = np.arange(lam)[::-1]

        selection = Block(0.5)
        p, e = selection(population, evals, sort=False)
        self.assertEqual(len(p), len(population))
        self.assertEqual(len(e), len(evals))
        expected = np.append(population[5:][::-1], population[5:][::-1])
        self.assertListEqual(list(expected), list(p))
        expected = np.append(evals[5:][::-1], evals[5:][::-1])
        self.assertEqual(list(expected), list(e))

        p, e = selection(population, evals, sort=True)
        self.assertEqual(len(p), len(population))
        self.assertEqual(len(e), len(evals))
        expected = [9, 9, 8, 8, 7, 7, 6, 6, 5, 5]
        self.assertListEqual(list(expected), list(p))
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        self.assertEqual(list(expected), list(e))

        lam = 7
        population = np.arange(lam)
        evals = np.arange(lam)[::-1]
        selection = Block(0.3)
        p, e = selection(population, evals, sort=False)
        self.assertEqual(len(p), len(population))
        self.assertEqual(len(e), len(evals))
        expected = [6, 5, 6, 5, 6, 5, 6]
        self.assertListEqual(list(expected), list(p))
        expected = [0, 1, 0, 1, 0, 1, 0]
        self.assertEqual(list(expected), list(e))

    def test_selection_2d(self):
        lam = 10
        population = np.random.randint(0, 1, (lam, 5))
        evals = np.arange(lam)[::-1]

        selection = Block(0.5)
        p, e = selection(population, evals, sort=False)
        self.assertEqual(len(p), len(population))
        self.assertEqual(len(e), len(evals))
        expected = np.vstack([population[5:][::-1], population[5:][::-1]])
        self.assertEqual(np.all(expected == p), True)
        expected = np.append(evals[5:][::-1], evals[5:][::-1])
        self.assertEqual(np.all(expected == e), True)

    def test_selection_3d(self):
        lam = 10
        population = np.random.randint(0, 1, (lam, 5, 2))
        evals = np.arange(lam)[::-1]

        selection = Block(0.5)
        p, e = selection(population, evals, sort=False)
        self.assertEqual(len(p), len(population))
        self.assertEqual(len(e), len(evals))
        expected = np.vstack([population[5:][::-1], population[5:][::-1]])
        self.assertEqual(np.all(expected == p), True)
        expected = np.append(evals[5:][::-1], evals[5:][::-1])
        self.assertEqual(np.all(expected == e), True)


if __name__ == "__main__":
    unittest.main()
